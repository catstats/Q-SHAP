# example:data
import numpy as np
from qshap import gazer

rng = np.random.default_rng(42)
x = rng.normal(size=(80, 4))
y = (
    2.0 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2] ** 2
    + rng.normal(scale=0.1, size=x.shape[0])
)
feature_names = np.array(["x0", "x1", "x2", "x3"])
# end:data

# example:model
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=8, max_depth=2, learning_rate=0.2, random_state=42
).fit(x, y)
# end:model

# example:compute
explainer = gazer(model)
contributions = explainer.rsq(x, y, progress_bar=False)
for index in np.argsort(contributions)[::-1]:
    print(f"{feature_names[index]}: {contributions[index]:.4f}")
# end:compute
