import numpy as np
import pytest

import qshap._backend as backend_module
import qshap.main as main_module
from qshap import gazer
from qshap.vis_module import vis


def test_gcorr_returns_square_root_and_is_static():
    rsq = np.array([0.0, 0.25, 1.0])
    expected = np.array([0.0, 0.5, 1.0])

    assert isinstance(gazer.__dict__["gcorr"], staticmethod)
    np.testing.assert_allclose(gazer.gcorr(rsq), expected)
    np.testing.assert_allclose(object.__new__(gazer).gcorr(rsq), expected)


def test_vis_gcorr_preserves_plotting_options(monkeypatch):
    captured = {}

    def fake_rsq(x, **kwargs):
        captured["x"] = x
        captured.update(kwargs)

    monkeypatch.setattr(vis, "rsq", staticmethod(fake_rsq))
    labels = np.array(["first", "second"])

    vis.gcorr(
        np.array([0.25, 1.0]),
        color_map_name="Reds",
        horizontal=True,
        max_feature=2,
        cutoff=0.1,
        title="Correlation",
        xtitle="Features",
        ytitle="GC",
        rotation=45,
        label=labels,
        decimal=4,
        save_name="gcorr_plot",
    )

    np.testing.assert_allclose(captured.pop("x"), np.array([0.5, 1.0]))
    assert captured["color_map_name"] == "Reds"
    assert captured["horizontal"] is True
    assert captured["model_rsq"] is False
    assert captured["max_feature"] == 2
    assert captured["cutoff"] == 0.1
    assert captured["title"] == "Correlation"
    assert captured["xtitle"] == "Features"
    assert captured["ytitle"] == "GC"
    assert captured["rotation"] == 45
    np.testing.assert_array_equal(captured["label"], labels)
    assert captured["decimal"] == 4
    assert captured["save_name"] == "gcorr_plot"


@pytest.mark.parametrize("ncore", [0, -2])
def test_resolve_ncore_rejects_invalid_integer_requests(ncore):
    with pytest.raises(ValueError, match="ncore"):
        main_module._resolve_ncore(ncore, n_samples=10)


@pytest.mark.parametrize("ncore", [True, 1.5, "2"])
def test_resolve_ncore_rejects_non_integer_requests(ncore):
    with pytest.raises(TypeError, match="ncore"):
        main_module._resolve_ncore(ncore, n_samples=10)


def test_resolve_ncore_caps_workers_to_samples_and_cpus(monkeypatch):
    monkeypatch.setattr(main_module.os, "cpu_count", lambda: 8)

    assert main_module._resolve_ncore(20, n_samples=3) == 3
    assert main_module._resolve_ncore(-1, n_samples=20) == 8


def test_resolve_ncore_handles_unknown_cpu_count(monkeypatch):
    monkeypatch.setattr(main_module.os, "cpu_count", lambda: None)

    assert main_module._resolve_ncore(-1, n_samples=10) == 1


def test_resolve_ncore_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one sample"):
        main_module._resolve_ncore(1, n_samples=0)


def test_backend_fallback_reports_and_preserves_import_failure(monkeypatch):
    import_error = ImportError("broken compiled extension")
    monkeypatch.setattr(backend_module, "_qshap_cpp", None)
    monkeypatch.setattr(backend_module, "_cpp_import_error", import_error)

    with pytest.warns(RuntimeWarning, match="Falling back"):
        assert backend_module.should_use_cpp("auto") is False

    with pytest.raises(RuntimeError, match="broken compiled extension") as exc_info:
        backend_module.should_use_cpp("cpp")
    assert exc_info.value.__cause__ is import_error
