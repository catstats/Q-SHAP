"""Exercise TreeSHAP and both Q-SHAP backends on supported regressors."""

import numpy as np
import pytest
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from qshap import gazer
from qshap._backend import cpp_available


@pytest.fixture(params=["tree", "gbdt", "xgboost", "lightgbm", "catboost"])
def fitted_regressor(request):
    rng = np.random.default_rng(20261018)
    X = rng.normal(size=(64, 4))
    y = 2 * X[:, 0] - X[:, 1] + 0.4 * X[:, 2] * X[:, 3]

    if request.param == "tree":
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
    elif request.param == "gbdt":
        model = GradientBoostingRegressor(
            n_estimators=5, max_depth=2, random_state=42
        )
    elif request.param == "xgboost":
        xgb = pytest.importorskip("xgboost")
        model = xgb.XGBRegressor(
            n_estimators=5, max_depth=2, n_jobs=1, random_state=42
        )
    elif request.param == "lightgbm":
        lgb = pytest.importorskip("lightgbm")
        model = lgb.LGBMRegressor(
            n_estimators=5, max_depth=2, num_leaves=4,
            min_child_samples=2, n_jobs=1, verbosity=-1, random_state=42
        )
    else:
        cb = pytest.importorskip("catboost")
        model = cb.CatBoostRegressor(
            iterations=5, depth=2, thread_count=1, random_seed=42,
            verbose=False, allow_writing_files=False
        )

    return model.fit(X, y), X, y


def test_treeshap_reconstructs_model_predictions(fitted_regressor):
    model, X, _ = fitted_regressor
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    assert values.shape == X.shape
    np.testing.assert_allclose(
        values.sum(axis=1) + explainer.expected_value,
        model.predict(X),
        atol=1e-6, rtol=1e-6,
    )


def test_cpp_and_numba_local_decompositions_agree(fitted_regressor):
    if not cpp_available():
        pytest.skip("requires an installed C++ extension")

    model, X, y = fitted_regressor
    explainer = gazer(model)
    cpp = explainer.rsq(X, y, local=True, backend="cpp", progress_bar=False)
    numba = explainer.rsq(X, y, local=True, backend="numba", progress_bar=False)

    for field in ("rsq", "loss", "local_rsq"):
        np.testing.assert_allclose(
            getattr(cpp, field), getattr(numba, field), atol=1e-8, rtol=1e-8
        )
    np.testing.assert_allclose(cpp.local_rsq.sum(axis=0), cpp.rsq)
