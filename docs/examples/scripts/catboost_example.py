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
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=8, depth=2, learning_rate=0.2, loss_function="RMSE",
    random_seed=42, thread_count=1, verbose=False,
    allow_writing_files=False,
).fit(x, y)
# end:model

# example:compute
explainer = gazer(model)
contributions = explainer.rsq(x, y, progress_bar=False)
for index in np.argsort(contributions)[::-1]:
    print(f"{feature_names[index]}: {contributions[index]:.4f}")
# end:compute
