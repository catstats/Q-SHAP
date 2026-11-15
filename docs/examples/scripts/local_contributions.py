# example:setup
import numpy as np
from qshap import gazer

rng = np.random.default_rng(42)
x = rng.normal(size=(80, 4))
y = (
    2.0 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2] ** 2
    + rng.normal(scale=0.1, size=x.shape[0])
)
feature_names = np.array(["x0", "x1", "x2", "x3"])

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=8, max_depth=2, learning_rate=0.2, random_state=42
).fit(x, y)

import matplotlib.pyplot as plt
from qshap import vis
# end:setup

# example:local
result = gazer(model).rsq(x, y, local=True, progress_bar=False)
print(result.rsq.shape)        # One value per feature.
print(result.local_rsq.shape)  # One row per observation, one column per feature.
print(result.loss.shape)

sst = np.sum((y - np.mean(y)) ** 2)
np.testing.assert_allclose(result.local_rsq, -result.loss / sst)
np.testing.assert_allclose(result.local_rsq.sum(axis=0), result.rsq)
# end:local

# example:heatmap
observation_ids = np.array([f"sample_{i}" for i in range(len(y))])
figure = vis.heatmap(
    result,
    feature_names=feature_names,
    observation_ids=observation_ids,
    n_show=12,
    show=False,
)
figure.savefig("local_contributions.pdf", bbox_inches="tight")
plt.close(figure)
# end:heatmap

# example:selection
figure = vis.heatmap(
    result,
    feature_names=feature_names,
    observation_ids=observation_ids,
    samples=["sample_0", "sample_10", "sample_20"],
    show=False,
)
figure.savefig("selected_observations.pdf", bbox_inches="tight")
plt.close(figure)
# end:selection

# example:loss
figure = vis.heatmap(
    result,
    quantity="loss",
    feature_names=feature_names,
    n_show=12,
    show=False,
)
figure.savefig("loss_contributions.pdf", bbox_inches="tight")
plt.close(figure)
# end:loss
