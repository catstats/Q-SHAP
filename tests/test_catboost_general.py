import numpy as np
import pytest

cb = pytest.importorskip("catboost")

import qshap.main as main_module
from qshap import gazer
from qshap._backend import cpp_available
from qshap.boosting_importance import fast_catboost_qshap_r2
from qshap.catboost_backend import route_tree_leaf_nodes
from qshap.utils import catboost_formatter


def _fit_catboost(policy, X, y, *, nan_mode="Min"):
    params = dict(
        iterations=3,
        depth=3,
        learning_rate=0.25,
        loss_function="RMSE",
        grow_policy=policy,
        nan_mode=nan_mode,
        verbose=False,
        allow_writing_files=False,
        random_seed=20260805,
        thread_count=1,
    )
    if policy == "Lossguide":
        params["max_leaves"] = 6
    return cb.CatBoostRegressor(**params).fit(X, y)


def _parsed_predict(explainer, X):
    prediction = np.full(X.shape[0], explainer.base_score, dtype=np.float64)
    for tree in explainer.catboost_res:
        prediction += tree.value[route_tree_leaf_nodes(X, tree)]
    return prediction


@pytest.mark.parametrize(
    ("policy", "is_symmetric"),
    [
        ("SymmetricTree", True),
        ("Depthwise", False),
        ("Lossguide", False),
    ],
)
@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_all_catboost_grow_policies_match_native_predictions_with_missing_values(
    policy, is_symmetric, nan_mode
):
    rng = np.random.default_rng(20260805)
    X = rng.normal(size=(72, 5))
    X[::8, 1] = np.nan
    y = (
        1.8 * np.nan_to_num(X[:, 1], nan=1.5)
        - 0.7 * X[:, 3]
        + rng.normal(scale=0.08, size=X.shape[0])
    )
    X_eval = rng.normal(size=(180, 5))
    X_eval[::7, 1] = np.nan

    model = _fit_catboost(policy, X, y, nan_mode=nan_mode)
    explainer = gazer(model)

    first_tree = explainer.catboost_res[0]
    first_internal = int(np.flatnonzero(first_tree.children_left >= 0)[0])
    split_feature = int(first_tree.feature[first_internal])
    border = float(first_tree.threshold[first_internal])
    just_above_in_float64 = np.nextafter(border, np.inf)
    assert just_above_in_float64 > border
    assert np.float32(just_above_in_float64) == np.float32(border)
    boundary_rows = np.zeros((3, X.shape[1]), dtype=np.float64)
    boundary_rows[:, split_feature] = [
        np.nextafter(border, -np.inf),
        border,
        just_above_in_float64,
    ]
    X_eval = np.vstack([X_eval, boundary_rows])

    assert explainer.catboost_is_symmetric is is_symmetric
    active_defaults = np.concatenate(
        [tree.default_left[tree.children_left >= 0] for tree in explainer.catboost_res]
    )
    if nan_mode == "Max":
        assert np.any(~active_defaults)
    else:
        assert np.all(active_defaults)
    np.testing.assert_allclose(
        _parsed_predict(explainer, X_eval),
        model.predict(X_eval),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide"])
def test_non_symmetric_catboost_falls_back_to_generic_qshap(
    policy, monkeypatch
):
    rng = np.random.default_rng(20260806)
    X = rng.normal(size=(36, 4))
    X[::6, 0] = np.nan
    y = (
        1.7 * np.nan_to_num(X[:, 0], nan=1.25)
        - 0.8 * X[:, 2]
        + 0.3 * X[:, 1] * X[:, 3]
        + rng.normal(scale=0.08, size=X.shape[0])
    )
    model = _fit_catboost(policy, X, y, nan_mode="Max")
    explainer = gazer(model)
    local = explainer.rsq(
        X, y, loss_out=True, progress_bar=False, backend="numba"
    )

    def fail_if_fast_path_is_used(*args, **kwargs):
        raise AssertionError("non-symmetric CatBoost entered the symmetric fast path")

    monkeypatch.setattr(
        main_module, "fast_catboost_qshap_r2", fail_if_fast_path_is_used
    )
    result = explainer.rsq(
        X, y, loss_out=False, progress_bar=False, backend="numba"
    )
    result_auto = explainer.rsq(
        X, y, loss_out=False, progress_bar=False, backend="auto"
    )

    np.testing.assert_allclose(result, local.rsq, atol=1e-10, rtol=0)
    np.testing.assert_allclose(result_auto, result, atol=1e-10, rtol=0)
    assert np.all(np.isfinite(result))
    # CatBoost's regularized leaf estimates can leave a very small
    # intercept-only remainder; the feature decomposition should still agree
    # with fitted R2 to the same scale as the existing symmetric backend.
    assert np.sum(result) == pytest.approx(model.score(X, y), abs=2e-3)


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide"])
def test_fast_catboost_api_rejects_complete_non_symmetric_trees(policy):
    rng = np.random.default_rng(13)
    X = rng.normal(size=(300, 5))
    y = (
        X[:, 0] * X[:, 1]
        + X[:, 2] * X[:, 3]
        + 0.2 * rng.normal(size=X.shape[0])
    )
    params = dict(
        iterations=2,
        depth=2,
        learning_rate=0.2,
        grow_policy=policy,
        verbose=False,
        allow_writing_files=False,
        random_seed=3,
        thread_count=1,
    )
    if policy == "Lossguide":
        params["max_leaves"] = 4
    explainer = gazer(cb.CatBoostRegressor(**params).fit(X, y))

    assert all(
        tree.node_count == (1 << (tree.max_depth + 1)) - 1
        for tree in explainer.catboost_res
    )
    with pytest.raises(NotImplementedError, match="SymmetricTree"):
        fast_catboost_qshap_r2(explainer, X, y, compute_sd=False)


def test_symmetric_fast_path_respects_catboost_missing_direction():
    assert cpp_available()

    rng = np.random.default_rng(20260807)
    X = rng.normal(size=(48, 4))
    X[::6, 0] = np.nan
    y = 2.0 * np.nan_to_num(X[:, 0], nan=1.25) - 0.4 * X[:, 2]
    model = _fit_catboost("SymmetricTree", X, y, nan_mode="Max")
    explainer = gazer(model)

    fast = explainer.rsq(X, y, progress_bar=False)
    generic = explainer.rsq(
        X, y, loss_out=True, progress_bar=False, backend="numba"
    ).rsq

    np.testing.assert_allclose(fast, generic, atol=1e-8, rtol=0)


def test_catboost_formatter_maps_float_indices_to_flat_columns():
    model_data = {
        "features_info": {
            "float_features": [
                {
                    "feature_index": 0,
                    "flat_feature_index": 2,
                    "nan_value_treatment": "AsTrue",
                }
            ]
        },
        "scale_and_bias": [1.7, [-0.3]],
        "trees": [
            {
                "left": {"value": -1.0, "weight": 2.5},
                "right": {"value": 2.0, "weight": 1.5},
                "split": {
                    "border": 0.5,
                    "float_feature_index": 0,
                    "split_type": "FloatFeature",
                },
            }
        ],
    }

    trees, bias, max_depth = catboost_formatter(model_data)

    assert bias == -0.3
    assert max_depth == 1
    assert trees[0].feature[0] == 2
    assert not trees[0].default_left[0]
    np.testing.assert_allclose(trees[0].n_node_samples, [4.0, 2.5, 1.5])
    predictions = bias + trees[0].value[
        route_tree_leaf_nodes(
            np.array([[0.0, 0.0, 0.25], [0.0, 0.0, 0.75]]),
            trees[0],
        )
    ]
    np.testing.assert_allclose(predictions, [-2.0, 3.1])


def test_catboost_formatter_rejects_categorical_splits_explicitly():
    model_data = {
        "trees": [
            {
                "left": {"value": -1.0, "weight": 2.0},
                "right": {"value": 1.0, "weight": 2.0},
                "split": {
                    "split_index": 0,
                    "split_type": "OnlineCtr",
                },
            }
        ]
    }

    with pytest.raises(NotImplementedError, match="FloatFeature"):
        catboost_formatter(model_data)


def test_zero_leaf_covers_receive_only_negligible_mass_and_keep_values():
    split = {
        "border": 0.0,
        "float_feature_index": 0,
        "split_type": "FloatFeature",
    }
    symmetric_data = {
        "oblivious_trees": [
            {
                "leaf_values": [7.0, -3.0],
                "leaf_weights": [0.0, 100.0],
                "splits": [split],
            }
        ]
    }
    general_data = {
        "trees": [
            {
                "left": {"value": 7.0, "weight": 0.0},
                "right": {"value": -3.0, "weight": 100.0},
                "split": split,
            }
        ]
    }

    for model_data in (symmetric_data, general_data):
        tree = catboost_formatter(model_data)[0][0]
        leaves = np.flatnonzero(tree.children_left < 0)
        zero_cover_leaf = leaves[np.argmax(tree.value[leaves])]
        assert tree.n_node_samples[zero_cover_leaf] == pytest.approx(1e-10)
        assert tree.n_node_samples[zero_cover_leaf] < 1e-6
        predictions = tree.value[
            route_tree_leaf_nodes(np.array([[-1.0], [1.0]]), tree)
        ]
        np.testing.assert_array_equal(predictions, [7.0, -3.0])


def test_catboost_formatter_rejects_missing_float_feature_metadata():
    model_data = {
        "features_info": {
            "float_features": [
                {"feature_index": 0, "flat_feature_index": 0}
            ]
        },
        "trees": [
            {
                "left": {"value": -1.0, "weight": 1.0},
                "right": {"value": 1.0, "weight": 1.0},
                "split": {
                    "border": 0.0,
                    "float_feature_index": 1,
                    "split_type": "FloatFeature",
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="no matching feature"):
        catboost_formatter(model_data)


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide"])
def test_wide_general_catboost_uses_and_explains_high_index_features(policy):
    rng = np.random.default_rng(20260811)
    n_features = 100
    X = rng.normal(size=(400, n_features))
    coefficients = np.linspace(0.5, 1.0, n_features)
    y = X @ coefficients + 0.05 * rng.normal(size=X.shape[0])
    params = dict(
        iterations=1,
        depth=6,
        learning_rate=0.2,
        grow_policy=policy,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        random_seed=20260811,
        random_strength=0,
        thread_count=1,
    )
    if policy == "Lossguide":
        params["max_leaves"] = 64

    model = cb.CatBoostRegressor(**params).fit(X, y)
    explainer = gazer(model)
    used_features = np.unique(
        np.concatenate(
            [
                tree.feature[tree.children_left >= 0]
                for tree in explainer.catboost_res
            ]
        )
    )

    assert explainer.max_depth == 6
    assert used_features.size > 20
    assert np.any(used_features >= 90)

    X_eval = rng.normal(size=(10, n_features))
    y_eval = X_eval @ coefficients
    np.testing.assert_allclose(
        _parsed_predict(explainer, X_eval),
        model.predict(X_eval),
        atol=1e-10,
        rtol=0,
    )

    result_auto = explainer.rsq(
        X_eval, y_eval, progress_bar=False, backend="auto"
    )
    result_numba = explainer.rsq(
        X_eval, y_eval, progress_bar=False, backend="numba"
    )
    assert result_auto.shape == (n_features,)
    assert np.all(np.isfinite(result_auto))
    assert np.all(np.isfinite(result_numba))
    np.testing.assert_allclose(result_auto, result_numba, atol=1e-10, rtol=0)
