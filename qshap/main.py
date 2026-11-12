import json
import os
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from numbers import Integral
from types import SimpleNamespace

import numpy as np
import shap
import sklearn.ensemble
import sklearn.tree
from tqdm.auto import tqdm

from qshap.boosting_importance import fast_catboost_qshap_r2
from qshap.catboost_backend import catboost_float_features, route_tree_leaf_nodes
from qshap.qshap import loss_treeshap
from qshap.utils import (
    catboost_formatter,
    divide_chunks,
    lgb_formatter,
    lgb_shap,
    simple_trees_to_shap_models,
    store_complex_root,
    store_complex_v_invc,
    summarize_tree,
    xgb_formatter,
)

try:
    import xgboost
except ImportError:  # pragma: no cover - depends on optional extras
    xgboost = None

try:
    import lightgbm
except ImportError:  # pragma: no cover - depends on optional extras
    lightgbm = None

try:
    import catboost
except ImportError:  # pragma: no cover - depends on optional extras
    catboost = None


def _is_xgboost_regressor(model):
    return xgboost is not None and isinstance(model, xgboost.sklearn.XGBRegressor)


def _is_lightgbm_regressor(model):
    return lightgbm is not None and isinstance(model, lightgbm.sklearn.LGBMRegressor)


def _is_catboost_regressor(model):
    return catboost is not None and isinstance(model, catboost.CatBoostRegressor)


def _supported_models_message():
    return (
        "Supported models are: scikit-learn DecisionTreeRegressor, "
        "scikit-learn GradientBoostingRegressor, XGBoost XGBRegressor "
        "(install with `pip install qshap[xgboost]`), LightGBM LGBMRegressor "
        "(install with `pip install qshap[lightgbm]`), and CatBoostRegressor "
        "(install with `pip install qshap[catboost]`)."
    )


def _resolve_ncore(ncore, n_samples):
    """Validate a worker request and cap it to useful available workers."""
    if isinstance(ncore, bool) or not isinstance(ncore, Integral):
        raise TypeError("ncore must be an integer or -1 to use all available cores")
    if n_samples < 1:
        raise ValueError("x must contain at least one sample")

    available_cores = os.cpu_count() or 1
    if ncore == -1:
        requested_cores = available_cores
    elif ncore < 1:
        raise ValueError("ncore must be a positive integer or -1")
    else:
        requested_cores = int(ncore)

    return min(requested_cores, available_cores, n_samples)


def _save_model_json(model, *, package_name):
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    model_filename = tmp.name
    tmp.close()
    try:
        if package_name == "catboost":
            model.save_model(model_filename, format="json")
        else:
            model.save_model(model_filename)

        with open(model_filename, "r") as file:
            return json.load(file)
    finally:
        if os.path.exists(model_filename):
            os.remove(model_filename)

class gazer:
    """Prepare a fitted regression tree model for Q-SHAP.

    Parameters
    ----------
    model : fitted regressor
        A scikit-learn DecisionTreeRegressor or GradientBoostingRegressor,
        XGBoost XGBRegressor, LightGBM LGBMRegressor, or CatBoostRegressor.
        Optional model libraries must be installed separately or through extras.
        Raw XGBoost Booster and classifier objects are not supported by this API.

    Raises
    ------
    NotImplementedError
        If the fitted model or a model's split representation is unsupported.
    ValueError
        If model parsing encounters invalid or unsupported metadata.

    Notes
    -----
    The explainer caches parsed trees and numerical preparation for reuse.
    CatBoost accepts numeric scalar regression trees using SymmetricTree,
    Depthwise, or Lossguide. Categorical/CTR, text, and embedding split
    representations are not supported. Input columns must retain training order.
    """
    def __init__(self, model):
        self.model = model
        self.explainer = None
        self.model_kind = None

        if isinstance(model, sklearn.tree.DecisionTreeRegressor):
            self.model_kind = "sklearn_tree"
            self.explainer = shap.TreeExplainer(model)
            self.max_depth = model.tree_.max_depth

        elif isinstance(model, sklearn.ensemble.GradientBoostingRegressor):
            self.model_kind = "sklearn_gbdt"
            self.explainer = shap.TreeExplainer(model)
            self.max_depth = model.max_depth

        elif _is_xgboost_regressor(model):
            self.model_kind = "xgboost"
            max_depth_xgb = model.get_params().get("max_depth")
            self.max_depth = max_depth_xgb if max_depth_xgb is not None and max_depth_xgb > 0 else 6

            model_data = _save_model_json(model, package_name="xgboost")
            base_score = model_data["learner"]["learner_model_param"]["base_score"]
            if isinstance(base_score, str):
                base_score = base_score.replace("[", "").replace("]", "")
            elif isinstance(base_score, list):
                base_score = base_score[0]
            self.base_score = np.float64(base_score)
            self.xgb_res = xgb_formatter(model_data, self.max_depth)

        elif _is_lightgbm_regressor(model):
            self.model_kind = "lightgbm"
            max_depth_lgb = model.get_params().get("max_depth")
            self.max_depth = max_depth_lgb if max_depth_lgb is not None and max_depth_lgb > 0 else 31
            self.lgb_res = lgb_formatter(model.booster_.trees_to_dataframe(), self.max_depth)
            self.lgb_shap_res = lgb_shap(self.lgb_res)

        elif _is_catboost_regressor(model):
            self.model_kind = "catboost"
            model_data = _save_model_json(model, package_name="catboost")
            self.catboost_is_symmetric = "oblivious_trees" in model_data
            self.catboost_res, self.base_score, self.max_depth = catboost_formatter(model_data)
            self.catboost_shap_res = simple_trees_to_shap_models(self.catboost_res)

        else:
            raise NotImplementedError(f"Model not supported yet. {_supported_models_message()}")

        # store v_inc * c /d evaluated at complex roots
        self.store_v_invc = store_complex_v_invc(self.max_depth * 2)
        self.store_z = store_complex_root(self.max_depth * 2)
        
    def get_tree(self, tree=0):
        """Return the parsed fields of one tree.

        Parameters
        ----------
        tree : int, optional
            Zero-based tree index (default 0). Python indexing also accepts negative
            indices; R's ``get_tree`` instead uses one-based positive indices.

        Returns
        -------
        dict
            ``children_left``, ``children_right`` (zero-based children, -1 at leaves),
            ``feature`` (zero-based columns, -1 at leaves), ``threshold``,
            ``max_depth``, ``n_node_samples`` (node cover/weight), ``value``,
            ``node_count``, ``default_left`` (missing routing, possibly None),
            and ``xgboost_split`` (strict float32 split convention).

        Raises
        ------
        IndexError
            If the tree index is out of range.
        TypeError
            If the tree index cannot be used for sequence indexing.
        """
        if self.model_kind == "sklearn_tree":
            trees = [self.model.tree_]
        elif self.model_kind == "sklearn_gbdt":
            trees = [estimator.tree_ for estimator in self.model.estimators_.ravel()]
        elif self.model_kind == "xgboost":
            trees = self.xgb_res
        elif self.model_kind == "lightgbm":
            trees = self.lgb_res
        else:
            trees = self.catboost_res
        parsed_tree = trees[tree]
        fields = (
            "children_left", "children_right", "feature", "threshold",
            "max_depth", "n_node_samples", "value", "node_count",
            "default_left", "xgboost_split",
        )
        result = {name: getattr(parsed_tree, name) for name in fields[:8]}
        result["default_left"] = getattr(parsed_tree, "default_left", None)
        result["xgboost_split"] = getattr(parsed_tree, "xgboost_split", False)
        return result


    def loss(self, x, y, y_mean_ori=None, progress_bar=True, backend="auto"):
        """Compute observation-level contributions to the change in squared loss.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_samples, n_features)
            Numeric feature matrix in the fitted model's column order.
        y : numpy.ndarray of shape (n_samples,)
            Numeric responses corresponding to the rows of ``x``.
        y_mean_ori : float or None, optional
            Original response mean. Defaults to ``mean(y)``; parallel chunks share
            the mean of the full evaluated sample.
        progress_bar : bool, optional
            Display progress over trees where supported. Defaults to True.
        backend : {"auto", "cpp", "numba"}, optional
            General-tree quadratic-term implementation. ``auto`` uses the compiled
            extension when available and warns before falling back to Numba.
            ``cpp`` requires the extension; ``numba`` uses the reference backend.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features)
            Raw loss contributions. Negating and dividing by the full-sample
            ``sum((y - mean(y)) ** 2)`` gives contributions to global R-squared.
            These are not per-observation R-squared values.

        Raises
        ------
        ValueError
            If a backend name or supported backend input is invalid.
        RuntimeError
            If ``backend="cpp"`` is requested but the extension cannot load.
        """
        max_depth = self.max_depth
        model = self.model
        store_v_invc = self.store_v_invc 
        store_z = self.store_z
        explainer = self.explainer

        if y_mean_ori is None:
            y_mean_ori = np.mean(y)

        if self.model_kind == "sklearn_tree":
            summary_tree = summarize_tree(model.tree_)
            loss = loss_treeshap(x, y, summary_tree, store_v_invc, store_z, explainer, backend=backend)

        # GBM 
        elif self.model_kind == "sklearn_gbdt":
            ensemble_tree = model.estimators_
            num_tree = len(model)
            staged_predict = list(model.staged_predict(x))
            # learning_rate 
            alpha = model.learning_rate
        
            loss = np.zeros_like(x, dtype=np.float64)
            
            iterator = tqdm(range(num_tree)) if progress_bar else range(num_tree)

            for i in iterator:
                if i==0:
                    res = y - y_mean_ori
                else:
                    res = y - staged_predict[i-1]
                    
                summary_tree = summarize_tree(ensemble_tree[i, 0].tree_)
                explainer = shap.TreeExplainer(ensemble_tree[i, 0])
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, alpha, backend=backend)

        # XGBOOST 
        elif self.model_kind == "xgboost":

            xgb_booster = model.get_booster()           
            xgb_res = self.xgb_res
            num_tree = len(xgb_res)

            warnings.filterwarnings("ignore", module="xgb")

            loss = np.zeros_like(x, dtype=np.float64)
            cumulative_prediction = np.full(x.shape[0], self.base_score, dtype=np.float64)
            xgb_dmatrix = xgboost.DMatrix(x)
            
            iterator = tqdm(range(num_tree)) if progress_bar else range(num_tree)

            for i in iterator:
                # get summary_tree first 
                res = y - cumulative_prediction
                
                summary_tree = summarize_tree(xgb_res[i])
                tree_booster = xgb_booster[i]
                explainer = shap.TreeExplainer(tree_booster)
                
                # learning rate is different
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, 1, backend=backend)
                tree_pred = tree_booster.predict(xgb_dmatrix, output_margin=True) - self.base_score
                cumulative_prediction += tree_pred

        # LightGBM
        elif self.model_kind == "lightgbm":
            lgb_res = self.lgb_res
            lgb_shap_res = self.lgb_shap_res
            num_tree = model.n_iter_

            loss = np.zeros_like(x, dtype=np.float64)
            cumulative_prediction = np.zeros(x.shape[0], dtype=np.float64)

            iterator = tqdm(range(num_tree)) if progress_bar else range(num_tree)

            for i in iterator:
                # get summary_tree first 
                res = y - cumulative_prediction
                
                summary_tree = summarize_tree(lgb_res[i])
                explainer = shap.TreeExplainer(lgb_shap_res[i])
                
                # learning rate is different
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, 1, backend=backend)
                tree_pred = model.booster_.predict(
                    x, start_iteration=i, num_iteration=1, raw_score=True
                )
                cumulative_prediction += tree_pred

        # CatBoost
        elif self.model_kind == "catboost":
            x = catboost_float_features(x)
            cb_res = self.catboost_res
            cb_shap_res = self.catboost_shap_res
            num_tree = len(cb_res)

            loss = np.zeros_like(x, dtype=np.float64)
            cumulative_prediction = np.full(
                x.shape[0], self.base_score, dtype=np.float64
            )

            iterator = tqdm(range(num_tree)) if progress_bar else range(num_tree)

            for i in iterator:
                res = y - cumulative_prediction

                summary_tree = summarize_tree(cb_res[i])
                explainer = shap.TreeExplainer(cb_shap_res[i])
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, 1, backend=backend)
                leaf_nodes = route_tree_leaf_nodes(x, cb_res[i])
                cumulative_prediction += cb_res[i].value[leaf_nodes]

        return loss
    

    def rsq(self, x, y, loss_out=False, ncore=1, nsample=None, nfrac=None,
            random_state=42, progress_bar=True, backend="auto", local=None):
        """Decompose the fitted model's R-squared into feature contributions.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_samples, n_features)
            Numeric feature matrix in the fitted model's column order.
        y : numpy.ndarray of shape (n_samples,)
            Numeric responses corresponding to the rows of ``x``.
        loss_out : bool, optional
            Legacy spelling for requesting local output. Defaults to False.
        ncore : int, optional
            Worker processes, default 1; -1 requests available CPUs. Capped by CPU
            count and sample size. Start scripts under a ``__main__`` guard.
        nsample : int or None, optional
            Evaluate a random subset of this size, strictly between 0 and the
            original row count. Takes precedence over ``nfrac``.
        nfrac : float or None, optional
            Subsample fraction strictly between 0 and 1; must select at least one row.
        random_state : int, optional
            NumPy random seed for subsampling, default 42.
        progress_bar : bool, optional
            Display tree progress for serial execution or completed chunks for
            parallel execution. Defaults to True. Worker startup precedes completion.
        backend : {"auto", "cpp", "numba"}, optional
            General-tree quadratic-term implementation. ``auto`` uses the compiled
            extension when available and warns before falling back to Numba.
            ``cpp`` requires the extension; ``numba`` uses the reference backend.
        local : bool or None, optional
            Alias overriding ``loss_out`` when supplied. ``True`` returns global
            values and both local matrices; defaults to None.

        Returns
        -------
        numpy.ndarray of shape (n_features,) or types.SimpleNamespace
            Global feature R-squared values when local output is disabled. Otherwise,
            the namespace has ``rsq`` (feature vector), ``loss`` (raw n-by-p loss
            contributions), and ``local_rsq = -loss / SST`` (n-by-p). Column sums
            of ``local_rsq`` equal ``rsq``. Row order is preserved without sampling;
            with sampling it follows the sampled rows.

        Raises
        ------
        ValueError
            If the response has zero variance, a sampling size/fraction is invalid,
            the sample is empty, or ``ncore`` is not positive or -1.
        TypeError
            If ``ncore`` is not an integer (booleans are rejected).
        RuntimeError
            If a required compiled backend cannot load.

        Notes
        -----
        Numeric symmetric CatBoost models use the specialized cached global
        algorithm when local output is disabled. This path is independent of
        ``backend`` and ``ncore``. Local output uses the matrix-valued path.
        Global feature contributions can be negative; their sum is the model's
        R-squared on the evaluated sample. Parallel processes can cost more than
        serial computation for small datasets.
        """

        if local is not None:
            loss_out = bool(local)
        
        if nsample is not None:
            if nsample <=0 or nsample >= x.shape[0]:
                raise ValueError("Samping sample size (nsample) must be larger than 0 and smaller than the total number of samples, use None for no sampling.")
            np.random.seed(random_state)
            sample_ind = np.random.choice(len(x), nsample, replace=False)
            x = x[sample_ind]
            y = y[sample_ind]
   
        if nfrac is not None and nsample is None:
            if nfrac <= 0 or nfrac >= 1:
                raise ValueError("Sample fraction (nfrac) must be between (0, 1), use None for no sampling.")
            np.random.seed(random_state)
            nsample = int(len(x) * nfrac)
            sample_ind = np.random.choice(len(x), nsample, replace=False)
            x = x[sample_ind]
            y = y[sample_ind]
                  
        ncore = _resolve_ncore(ncore, len(x))

        explainer = self.explainer
        y_mean_ori = np.mean(y)
        sst = np.sum((y - y_mean_ori) ** 2)

        if sst <= 0:
            raise ValueError("Cannot compute R2 decomposition when y has zero variance")

        if (
            self.model_kind == "catboost"
            and self.catboost_is_symmetric
            and not loss_out
        ):
            return fast_catboost_qshap_r2(self, x, y, compute_sd=False)
        
        if ncore==1:
            loss = self.loss(x, y, y_mean_ori=y_mean_ori, progress_bar=progress_bar, backend=backend)
        else:
            x_chunks = divide_chunks(x, ncore)
            y_chunks = divide_chunks(y, ncore)

            # Display progress before starting workers. Collect in completion
            # order, but concatenate in input order so local rows stay aligned.
            results = [None] * ncore
            with tqdm(total=ncore, desc="Q-SHAP chunks", unit="chunk",
                      disable=not progress_bar) as progress:
                with ProcessPoolExecutor(max_workers=ncore) as executor:
                    futures = {
                        executor.submit(self.loss, x_chunks[i], y_chunks[i],
                                        y_mean_ori, False, backend): i
                        for i in range(ncore)
                    }
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()
                        progress.update(1)
            
            loss = np.concatenate(results) 

        local_rsq = -loss / sst
        rsq = np.sum(local_rsq, axis=0)

        if loss_out:
            return SimpleNamespace(rsq=rsq, loss=loss, local_rsq=local_rsq)
        else:
            return rsq
        

    @staticmethod
    def gcorr(rsq_res):
        """Take the square root of feature R-squared contributions.

        Parameters
        ----------
        rsq_res : numpy.ndarray of shape (n_features,)
            Global contribution vector, not a local-output namespace.

        Returns
        -------
        numpy.ndarray
            Elementwise square roots. Negative contributions produce NaN with the
            NumPy invalid-value warning; they are not clipped.
        """
        return np.sqrt(rsq_res)
