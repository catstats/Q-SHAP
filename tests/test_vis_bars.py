import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from qshap import vis


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("values", [[-2., 1., 0.], [-3., -1.], [0., 0.], [2., 2.]])
def test_signed_and_constant_bars_remain_visible(monkeypatch, horizontal, values):
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        vis.rsq(np.array(values), cutoff=-np.inf, horizontal=horizontal)

    figure, = figures
    axis, = figure.axes
    FigureCanvasAgg(figure).draw()
    bars = axis.patches
    lengths = [bar.get_width() if horizontal else bar.get_height() for bar in bars]
    np.testing.assert_allclose(lengths, sorted(values, reverse=True))
    low, high = axis.get_xlim() if horizontal else axis.get_ylim()
    assert low <= min(values) <= max(values) <= high
    assert low < high
    assert all(np.all(np.isfinite(bar.get_facecolor())) for bar in bars)
    for text in axis.texts:
        bounds = text.get_window_extent(figure.canvas.get_renderer())
        assert figure.bbox.contains(*bounds.p0)
        assert figure.bbox.contains(*bounds.p1)


def test_loss_save_preserves_negative_values_without_a_notebook(monkeypatch, tmp_path):
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))

    def no_widget(*args, **kwargs):
        pytest.fail("Saving one observation should not construct a notebook widget")

    monkeypatch.setattr("qshap.vis_module.widgets.interactive_output", no_widget)
    vis.loss(np.array([[-2., -1.]]), save_ind=0, save_prefix=str(tmp_path / "loss"))
    figure, = figures
    assert sorted(bar.get_height() for bar in figure.axes[0].patches) == [-2., -1.]
    assert (tmp_path / "loss 0.pdf").stat().st_size > 1000


def test_rsq_reports_when_cutoff_hides_every_feature():
    with pytest.raises(ValueError, match="at least one value"):
        vis.rsq(np.array([-2., -1.]))
