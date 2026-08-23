from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qshap import vis


def _local_result():
    local_rsq = np.asarray(
        [
            [-0.30, -0.10, 0.00],
            [0.05, 0.15, 0.10],
            [-0.05, -0.04, -0.01],
            [0.20, 0.35, 0.25],
            [-0.40, -0.20, -0.10],
            [0.02, 0.03, 0.05],
        ]
    )
    return SimpleNamespace(
        rsq=np.sum(local_rsq, axis=0),
        loss=-10.0 * local_rsq,
        local_rsq=local_rsq,
    )


def test_heatmap_orders_features_and_selects_both_observation_extremes():
    result = _local_result()
    figure = vis.heatmap(
        result,
        feature_names=["first", "second", "third"],
        observation_ids=[f"id_{index}" for index in range(6)],
        n_show=4,
        show=False,
    )
    figure.canvas.draw()

    heatmap_axis, total_axis, colorbar_axis = figure.axes
    assert [tick.get_text() for tick in heatmap_axis.get_xticklabels()] == [
        "third",
        "second",
        "first",
    ]
    assert [tick.get_text() for tick in heatmap_axis.get_yticklabels()] == [
        "id_3",
        "id_1",
        "id_0",
        "id_4",
    ]
    assert [tick.get_text() for tick in total_axis.get_xticklabels()] == ["Total"]
    assert figure._suptitle.get_text() == (
        "Observation-level contributions to the global R² decomposition"
    )
    assert any("%" in tick.get_text() for tick in colorbar_axis.get_yticklabels())

    norm = heatmap_axis.images[0].norm
    assert norm.vmin == -norm.vmax
    np.testing.assert_allclose(
        np.asarray(total_axis.images[0].get_array()).ravel(),
        np.asarray([0.80, 0.30, -0.40, -0.70]),
    )
    plt.close(figure)


def test_heatmap_preserves_explicit_sample_order_and_accepts_ids():
    result = _local_result()
    figure = vis.heatmap(
        result,
        feature_names=["first", "second", "third"],
        observation_ids=[f"id_{index}" for index in range(6)],
        samples=["id_5", "id_2"],
        show=False,
    )

    heatmap_axis = figure.axes[0]
    assert [tick.get_text() for tick in heatmap_axis.get_yticklabels()] == [
        "id_5",
        "id_2",
    ]
    plt.close(figure)
