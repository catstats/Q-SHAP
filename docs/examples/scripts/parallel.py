import numpy as np
from xgboost import XGBRegressor
from qshap import gazer


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 3))
    y = x[:, 0] - x[:, 1]
    model = XGBRegressor(
        n_estimators=2, max_depth=2, random_state=42, n_jobs=1
    ).fit(x, y)
    explainer = gazer(model)

    parallel = explainer.rsq(x, y, local=True, ncore=2)
    serial = explainer.rsq(x, y, local=True, ncore=1, progress_bar=False)
    for field in ("rsq", "loss", "local_rsq"):
        np.testing.assert_allclose(
            getattr(parallel, field), getattr(serial, field),
            rtol=1e-8, atol=1e-10,
        )
    print("Global contributions:", parallel.rsq)
    print("Local matrix shape:", parallel.local_rsq.shape)


if __name__ == "__main__":
    main()
