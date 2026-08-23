import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from qshap import gazer
from qshap.utils import summarize_tree


def test_summarize_tree_accepts_current_sklearn_value_shape():
    X, y = make_regression(
        n_samples=60,
        n_features=5,
        n_informative=3,
        random_state=20260702,
    )
    model = DecisionTreeRegressor(max_depth=2, random_state=20260702).fit(X, y)

    summary = summarize_tree(model.tree_)

    assert summary.value.ndim == 1
    assert summary.value.shape == (model.tree_.node_count,)
    np.testing.assert_allclose(summary.value, np.squeeze(model.tree_.value))


def test_sklearn_decision_tree_rsq_smoke():
    X, y = make_regression(
        n_samples=80,
        n_features=6,
        n_informative=3,
        random_state=20260702,
    )
    model = DecisionTreeRegressor(max_depth=2, random_state=20260702).fit(X, y)

    rsq = gazer(model).rsq(X, y, progress_bar=False)

    assert rsq.shape == (X.shape[1],)
    assert np.all(np.isfinite(rsq))


def test_local_alias_returns_rsq_loss_and_local_rsq():
    X, y = make_regression(
        n_samples=40,
        n_features=4,
        n_informative=2,
        random_state=20260718,
    )
    model = DecisionTreeRegressor(max_depth=2, random_state=20260718).fit(X, y)

    result = gazer(model).rsq(X, y, local=True, progress_bar=False)

    assert result.rsq.shape == (X.shape[1],)
    assert result.loss.shape == X.shape
    assert result.local_rsq.shape == X.shape
    assert np.all(np.isfinite(result.loss))
    assert np.all(np.isfinite(result.local_rsq))
    sst = np.sum((y - np.mean(y)) ** 2)
    np.testing.assert_allclose(result.local_rsq, -result.loss / sst)
    np.testing.assert_allclose(
        np.sum(result.local_rsq, axis=0),
        result.rsq,
        atol=1e-12,
        rtol=1e-12,
    )
