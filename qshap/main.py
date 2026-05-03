import json
import os
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
import shap
import sklearn.ensemble
import sklearn.tree
from tqdm import tqdm

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
            self.catboost_res, self.base_score, self.max_depth = catboost_formatter(model_data)
            self.catboost_shap_res = simple_trees_to_shap_models(self.catboost_res)

        else:
            raise NotImplementedError(f"Model not supported yet. {_supported_models_message()}")

        # store v_inc * c /d evaluated at complex roots
        self.store_v_invc = store_complex_v_invc(self.max_depth * 2)
        self.store_z = store_complex_root(self.max_depth * 2)
        

    def loss(self, x, y, y_mean_ori=None, progress_bar=True, backend="auto"):
        """
        Parameters
        -x: x
        -y: y
        -y_mean_ori: mean of the original
        -progress_bar: whether show the progress bar or not
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
            
            iterator = tqdm(range(num_tree)) if progress_bar else range(num_tree)

            for i in iterator:
                # get summary_tree first 
                if i==0:
                    res = y - self.base_score
                else:
                    res = y - model.predict(x, iteration_range=(0, i))
                
                summary_tree = summarize_tree(xgb_res[i])
                explainer = shap.TreeExplainer(xgb_booster[i])
                
                # learning rate is different
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, 1, backend=backend)

        # LightGBM
        elif self.model_kind == "lightgbm":
            lgb_res = self.lgb_res
            lgb_shap_res = self.lgb_shap_res
            num_tree = model.n_iter_

            loss = np.zeros_like(x, dtype=np.float64)

            iterator = tqdm(range(num_tree)) if progress_bar else range(num_tree)

            for i in iterator:
                # get summary_tree first 
                if i==0:
                    res = y 
                else:
                    res = y - model.predict(x, num_iteration=i, raw_score=True)
                
                summary_tree = summarize_tree(lgb_res[i])
                explainer = shap.TreeExplainer(lgb_shap_res[i])
                
                # learning rate is different
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, 1, backend=backend)

        # CatBoost
        elif self.model_kind == "catboost":
            cb_res = self.catboost_res
            cb_shap_res = self.catboost_shap_res
            num_tree = len(cb_res)

            loss = np.zeros_like(x, dtype=np.float64)

            iterator = tqdm(range(num_tree)) if progress_bar else range(num_tree)

            for i in iterator:
                if i == 0:
                    res = y - self.base_score
                else:
                    res = y - model.predict(x, ntree_end=i)

                summary_tree = summarize_tree(cb_res[i])
                explainer = shap.TreeExplainer(cb_shap_res[i])
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, 1, backend=backend)

        return loss
    

    def rsq(self, x, y, loss_out=False, ncore=1, nsample=None, nfrac=None, random_state=42, progress_bar=True, backend="auto"):
        """
        Parameters
        -x: the original x
        -y: the original y
        -loss_out: output loss or not
        -nsample: number of samples to sample from, by default use all samples
        -nfrac: fraction of samples to sample from, by default 1, use all samples
        -ncore: number of cores to use, with default value 1. It will NOT be beneficial for small datasets and shallow depth.
        -random_state: control random seed for numpy
        -progress_bar: whether show the progress bar or not
        -backend: "auto" uses the compiled C++ backend when available; "numba"
         uses the Python/numba reference implementation for understanding the algorithm.

        Return
        Shapley R-squared
        """ 

        
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
                  
        max_core = os.cpu_count()
        if ncore == -1:
            ncore = os.cpu_count()
        ncore = min(max_core, ncore)

        explainer = self.explainer
        y_mean_ori = np.mean(y)
        sst = np.sum((y - y_mean_ori) ** 2)
        
        if ncore==1:
            loss = self.loss(x, y, y_mean_ori=y_mean_ori, progress_bar=progress_bar, backend=backend)
        else:
            x_chunks = divide_chunks(x, ncore)
            y_chunks = divide_chunks(y, ncore)

            with ProcessPoolExecutor(max_workers=ncore) as executor:
                # Submit all chunks for processing
                futures = [executor.submit(self.loss, x_chunks[i], y_chunks[i], y_mean_ori, False, backend) for i in range(ncore)]

                # Assign progress bar to a variable
                iterator = tqdm(futures, desc="Processing", total=len(futures)) if progress_bar else futures

                # Wait for all futures to complete and collect results
                results = [future.result() for future in iterator]
            
            loss = np.concatenate(results) 

        rsq = 0 - np.sum(loss, axis=0)/sst

        if loss_out:
            return SimpleNamespace(rsq=rsq, loss=loss)
        else:
            return rsq
        

    def gcorr(rsq_res):
        """
        Parameters
        -rsq_res: the rsq result from calling gazer.rsq

        Return
        Generalized correlation (Square root of Shapley R-squared)
        """
        res = np.sqrt(rsq_res)
        return rsq_res
