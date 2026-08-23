import json

import numpy as np
import pytest

cb = pytest.importorskip("catboost")

from qshap import gazer
from qshap._backend import catboost_qshap_r2_fast_cpp, cpp_available
from qshap.boosting_importance import fast_catboost_qshap_r2
from qshap.catboost_backend import route_oblivious_leaves
from qshap.catboost_cached_importance import _catboost_global_loss_stats_python


def test_catboost_fast_rsq_matches_local_backend():
    assert cpp_available()

    rng = np.random.default_rng(20260512)
    n = 160
    p = 12
    X = rng.normal(size=(n, p))
    y = 2 * X[:, 0] - 1.3 * X[:, 1] + 0.4 * X[:, 2] * X[:, 3] + rng.normal(scale=0.1, size=n)

    model = cb.CatBoostRegressor(
        iterations=12,
        depth=3,
        learning_rate=0.15,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        random_seed=20260512,
    ).fit(X, y)

    g = gazer(model)
    fast = g.rsq(X, y, loss_out=False)
    slow = g.rsq(X, y, loss_out=True).rsq

    parsed = np.full(n, g.base_score, dtype=np.float64)
    parsed_leaf_ids = []
    for tree in g.catboost_res:
        leaf_ids = route_oblivious_leaves(X, tree)
        parsed_leaf_ids.append(leaf_ids)
        num_internal = (1 << int(tree.max_depth)) - 1
        parsed += tree.value[num_internal + leaf_ids]

    native_leaf_ids = model.calc_leaf_indexes(X).astype(np.int64)
    np.testing.assert_array_equal(
        np.column_stack(parsed_leaf_ids), native_leaf_ids
    )
    np.testing.assert_allclose(parsed, model.predict(X), atol=1e-10, rtol=0)
    fused = catboost_qshap_r2_fast_cpp(
        X,
        y,
        g.catboost_res,
        g.base_score,
        compute_sd=False,
        return_prediction=True,
    )
    np.testing.assert_allclose(
        fused["prediction"], model.predict(X), atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(fast, slow, atol=1e-8, rtol=0)
    assert abs(np.sum(fast) - np.sum(slow)) < 1e-8


def test_catboost_fast_sd_matches_total_local_loss():
    assert cpp_available()

    rng = np.random.default_rng(20260808)
    n = 96
    p = 7
    X = rng.normal(size=(n, p))
    y = (
        1.8 * X[:, 0]
        - 0.9 * X[:, 2]
        + 0.5 * X[:, 1] * X[:, 4]
        + rng.normal(scale=0.12, size=n)
    )
    model = cb.CatBoostRegressor(
        iterations=9,
        depth=3,
        learning_rate=0.18,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        random_seed=20260808,
        thread_count=1,
    ).fit(X, y)
    explainer = gazer(model)

    local = explainer.rsq(
        X, y, loss_out=True, progress_bar=False, backend="numba"
    )
    expected_loss_sum = np.sum(local.loss, axis=0)
    expected_loss_sumsq = np.sum(local.loss * local.loss, axis=0)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    loss_var = np.maximum(
        (
            expected_loss_sumsq
            - expected_loss_sum * expected_loss_sum / n
        )
        / (n - 1),
        0.0,
    )
    expected_sd_rsq = np.sqrt(n * loss_var) / sst

    fast = fast_catboost_qshap_r2(explainer, X, y, compute_sd=True)
    python_stats = _catboost_global_loss_stats_python(
        explainer, X, y, compute_sd=True
    )
    python_no_sd = _catboost_global_loss_stats_python(
        explainer, X, y, compute_sd=False
    )

    np.testing.assert_allclose(fast.loss_sum, expected_loss_sum, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        fast.loss_sumsq, expected_loss_sumsq, atol=1e-8, rtol=1e-10
    )
    np.testing.assert_allclose(fast.sd_rsq, expected_sd_rsq, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(
        python_stats.loss_sum, expected_loss_sum, atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        python_stats.loss_sumsq,
        expected_loss_sumsq,
        atol=1e-8,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        python_no_sd.loss_sum, expected_loss_sum, atol=1e-8, rtol=0
    )
    assert python_no_sd.loss_sumsq is None


def test_catboost_fast_cpp_rejects_missing_input_columns_before_routing():
    assert cpp_available()

    rng = np.random.default_rng(20260810)
    X = rng.normal(size=(80, 3))
    y = 4.0 * X[:, 2] + rng.normal(scale=0.01, size=X.shape[0])
    model = cb.CatBoostRegressor(
        iterations=3,
        depth=2,
        learning_rate=0.3,
        verbose=False,
        allow_writing_files=False,
        random_seed=20260810,
        thread_count=1,
    ).fit(X, y)
    explainer = gazer(model)

    assert any(
        np.any(tree.feature[tree.children_left >= 0] >= 1)
        for tree in explainer.catboost_res
    )
    with pytest.raises(RuntimeError, match="out of bounds"):
        fast_catboost_qshap_r2(
            explainer, X[:, :1], y, compute_sd=False
        )


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_catboost_quantized_router_exactness_gates(nan_mode):
    rng = np.random.default_rng(20260817)
    X = rng.normal(size=(180, 6))
    X[rng.choice(X.shape[0], 28, replace=False), 0] = np.nan
    X[rng.choice(X.shape[0], 21, replace=False), 3] = np.nan
    y = (
        np.where(np.isnan(X[:, 0]), 0.8, X[:, 0])
        - 0.7 * X[:, 2]
        + 0.4 * np.where(np.isnan(X[:, 3]), -0.2, X[:, 3])
        + rng.normal(scale=0.05, size=X.shape[0])
    )

    model = cb.CatBoostRegressor(
        iterations=18,
        depth=4,
        learning_rate=0.12,
        loss_function="RMSE",
        nan_mode=nan_mode,
        ignored_features=[1, 4],
        verbose=False,
        allow_writing_files=False,
        random_seed=20260817,
    ).fit(X, y)
    g = gazer(model)

    parsed_leaf_ids = np.column_stack(
        [route_oblivious_leaves(X, tree) for tree in g.catboost_res]
    )
    native_leaf_ids = model.calc_leaf_indexes(X).astype(np.int64)
    np.testing.assert_array_equal(parsed_leaf_ids, native_leaf_ids)

    fused = catboost_qshap_r2_fast_cpp(
        X,
        y,
        g.catboost_res,
        g.base_score,
        compute_sd=False,
        return_prediction=True,
    )
    np.testing.assert_allclose(
        fused["prediction"], model.predict(X), atol=1e-10, rtol=0
    )

    fast = g.rsq(X, y, loss_out=False)
    slow = g.rsq(X, y, loss_out=True, progress_bar=False).rsq
    np.testing.assert_allclose(fast, slow, atol=1e-8, rtol=0)


def test_catboost_float32_boundary_and_scale_bias(tmp_path):
    X_train = np.linspace(-2.0, 2.0, 160, dtype=np.float64)[:, None]
    y_train = np.where(X_train[:, 0] > 0.25, 2.0, -1.0)
    model = cb.CatBoostRegressor(
        iterations=1,
        depth=1,
        learning_rate=0.3,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        random_seed=20260818,
    ).fit(X_train, y_train)
    g = gazer(model)

    border = float(np.float32(g.catboost_res[0].threshold[0]))
    same_float32 = np.nextafter(border, np.inf)
    next_float32 = float(
        np.nextafter(np.float32(border), np.float32(np.inf), dtype=np.float32)
    )
    X_test = np.array([[border], [same_float32], [next_float32]])
    y_test = np.array([-0.2, 0.4, 1.1])
    assert np.float32(same_float32) == np.float32(border)

    native_leaf = model.calc_leaf_indexes(X_test).astype(np.int64)[:, 0]
    parsed_leaf = route_oblivious_leaves(X_test, g.catboost_res[0])
    np.testing.assert_array_equal(parsed_leaf, native_leaf)
    assert native_leaf[0] == native_leaf[1]

    model_json = tmp_path / "scaled_catboost.json"
    model.save_model(str(model_json), format="json")
    model_data = json.loads(model_json.read_text())
    model_data["scale_and_bias"] = [1.7, [-0.3]]
    model_json.write_text(json.dumps(model_data))
    scaled_model = cb.CatBoostRegressor()
    scaled_model.load_model(str(model_json), format="json")
    scaled_g = gazer(scaled_model)
    fused = catboost_qshap_r2_fast_cpp(
        X_test,
        y_test,
        scaled_g.catboost_res,
        scaled_g.base_score,
        compute_sd=False,
        return_prediction=True,
    )
    np.testing.assert_allclose(
        fused["prediction"], scaled_model.predict(X_test), atol=1e-12, rtol=0
    )
