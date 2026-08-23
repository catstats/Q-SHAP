import numpy as np
import pytest

from qshap import gazer


def test_xgboost_boundaries_and_missing_directions_match_fitted_r2():
    xgboost = pytest.importorskip("xgboost")
    X = np.tile(np.array([[0.0], [1.0], [2.0], [np.nan]]), (50, 1))
    y = np.tile(np.array([0.0, 10.0, 20.0, -10.0]), 50)
    model = xgboost.XGBRegressor(
        n_estimators=5,
        max_depth=1,
        learning_rate=0.3,
        n_jobs=1,
        verbosity=0,
        random_state=42,
    ).fit(X, y)

    explainer = gazer(model)
    assert any(np.any(tree.default_left) for tree in explainer.xgb_res)
    for backend in ("auto", "numba"):
        qshap_r2 = np.sum(
            explainer.rsq(X, y, progress_bar=False, backend=backend)
        )
        assert qshap_r2 == pytest.approx(model.score(X, y), abs=1e-6)
