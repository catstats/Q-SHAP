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

from qshap import vis

result = gazer(model).rsq(x, y, local=True, progress_bar=False)
contributions = result.rsq
# end:setup

# example:bar
vis.rsq(
    contributions,
    label=feature_names,
    max_feature=len(feature_names),
    color_map_name="Blues",
    save_name="global_rsq",
)
# end:bar

# example:horizontal
vis.rsq(
    contributions,
    label=feature_names,
    max_feature=len(feature_names),
    horizontal=True,
    color_map_name="PuBu",
    save_name="global_rsq_horizontal",
)
# end:horizontal

# example:cumulative
vis.cumu(
    contributions,
    max_comp=len(feature_names),
    save_name="cumulative_rsq",
)
# end:cumulative

# example:elbow
selected = vis.elbow(contributions, max_comp=len(feature_names))
print("Displayed feature order:", feature_names[selected])
# end:elbow

# example:gcorr
if np.all(contributions >= 0):
    correlation = gazer.gcorr(contributions)
    print("Generalized correlations:", correlation)
    vis.gcorr(
        contributions,
        label=feature_names,
        max_feature=len(feature_names),
        save_name="generalized_correlation",
    )
else:
    print("The square-root transformation is undefined for negative contributions.")
# end:gcorr
