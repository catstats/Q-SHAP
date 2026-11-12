from concurrent.futures import Future

import numpy as np
import pytest

import qshap.main as main
from qshap import gazer


def test_parallel_completion_preserves_local_row_order(monkeypatch):
    events = []

    class Progress:
        def __init__(self, **kwargs):
            assert kwargs['total'] == 2
            events.append('progress')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, count):
            events.append('complete')

    class Executor:
        def __init__(self, max_workers):
            events.append('executor')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, x, y, mean, progress, backend):
            assert mean == 2.5  # Global mean, not the mean of a chunk.
            future = Future()
            future.set_result(x.copy())
            return future

    monkeypatch.setattr(main, 'tqdm', Progress)
    monkeypatch.setattr(main, 'ProcessPoolExecutor', Executor)
    monkeypatch.setattr(main, 'as_completed', lambda futures: reversed(list(futures)))
    monkeypatch.setattr(main.os, 'cpu_count', lambda: 2)
    explainer = object.__new__(gazer)
    explainer.explainer = None
    explainer.model_kind = 'test'
    x = np.arange(8.0).reshape(4, 2)
    y = np.arange(1.0, 5.0)
    result = explainer.rsq(x, y, ncore=2, local=True)
    np.testing.assert_array_equal(result.loss, x)
    np.testing.assert_allclose(result.local_rsq, -x / 5.0)
    np.testing.assert_allclose(result.rsq, result.local_rsq.sum(axis=0))
    assert events == ['progress', 'executor', 'complete', 'complete']


def test_real_parallel_matches_serial():
    xgb = pytest.importorskip('xgboost')
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 3))
    y = x[:, 0] - x[:, 1]
    model = xgb.XGBRegressor(n_estimators=2, max_depth=2, n_jobs=1).fit(x, y)
    explainer = gazer(model)
    serial = explainer.rsq(x, y, local=True, ncore=1, progress_bar=False)
    parallel = explainer.rsq(x, y, local=True, ncore=2, progress_bar=False)
    for field in ('loss', 'rsq', 'local_rsq'):
        np.testing.assert_allclose(getattr(serial, field), getattr(parallel, field),
                                   rtol=1e-8, atol=1e-10)
