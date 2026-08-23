"""Simple no-tuning Q-SHAP benchmark for XGBoost, LightGBM, and CatBoost.

The default setting uses simulation model b from the Q-SHAP paper:

    Y = 4 X1 - 5 X2 + 6 X3 + 3 X1 X2 - X1 X3 + eps

Only the first three features are real signal features. The script prints the
first five feature-specific R^2 values so features 4 and 5 can be checked as
near-zero nuisance controls.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qshap import gazer


SIGNAL_PROBS = np.array([0.6, 0.7, 0.5], dtype=np.float64)


def signal_function(X3: np.ndarray, model_name: str) -> np.ndarray:
    x1 = X3[:, 0]
    x2 = X3[:, 1]
    x3 = X3[:, 2]
    value = 4.0 * x1 - 5.0 * x2 + 6.0 * x3
    if model_name == "b":
        value = value + 3.0 * x1 * x2 - x1 * x3
    elif model_name == "c":
        value = value + 3.0 * x1 * x2 - x1 * x2 * x3
    elif model_name != "a":
        raise ValueError(f"Unknown simulation model: {model_name}")
    return value


def make_data(n: int, p: int, sigma: float, model_name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.empty((n, p), dtype=np.float64)
    X[:, 0] = rng.binomial(1, SIGNAL_PROBS[0], size=n)
    X[:, 1] = rng.binomial(1, SIGNAL_PROBS[1], size=n)
    X[:, 2] = rng.binomial(1, SIGNAL_PROBS[2], size=n)
    if p > 3:
        X[:, 3:] = rng.binomial(1, 0.5, size=(n, p - 3))
    y = signal_function(X[:, :3], model_name) + rng.normal(0.0, sigma, size=n)
    return X, y


def build_models(args: argparse.Namespace) -> dict[str, object]:
    models: dict[str, object] = {}
    if "xgboost" in args.models:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=args.trees,
            max_depth=args.depth,
            learning_rate=args.lr,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=args.seed,
            n_jobs=1,
        )
    if "lightgbm" in args.models:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = LGBMRegressor(
            n_estimators=args.trees,
            max_depth=args.depth,
            num_leaves=2**args.depth,
            learning_rate=args.lr,
            objective="regression",
            random_state=args.seed,
            n_jobs=1,
            verbose=-1,
        )
    if "catboost" in args.models:
        from catboost import CatBoostRegressor

        models["catboost"] = CatBoostRegressor(
            iterations=args.trees,
            depth=args.depth,
            learning_rate=args.lr,
            loss_function="RMSE",
            random_seed=args.seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    return models


def benchmark_one(name: str, model: object, X: np.ndarray, y: np.ndarray) -> None:
    start = time.perf_counter()
    model.fit(X, y)
    fit_sec = time.perf_counter() - start

    pred = model.predict(X)
    model_rsq = 1.0 - float(np.sum((y - pred) ** 2)) / float(np.sum((y - y.mean()) ** 2))

    start = time.perf_counter()
    explainer = gazer(model)
    explainer_sec = time.perf_counter() - start

    start = time.perf_counter()
    rsq = explainer.rsq(X, y, loss_out=False, progress_bar=False)
    qshap_sec = time.perf_counter() - start

    real5 = np.arange(5)
    print(f"\n{name}")
    print(f"fit_sec={fit_sec:.4f} explainer_sec={explainer_sec:.4f} qshap_sec={qshap_sec:.4f}")
    print(f"model_rsq={model_rsq:.8f} sum_qshap={np.sum(rsq):.8f}")
    print(f"real5_python_0based={real5.tolist()}")
    print(f"real5_R_1based={(real5 + 1).tolist()}")
    print(f"real5_R2={np.array2string(rsq[real5], precision=8, separator=', ')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--depth", "--d", dest="depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--simulation-model", choices=("a", "b", "c"), default="b")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("xgboost", "lightgbm", "catboost"),
        default=["xgboost", "lightgbm", "catboost"],
    )
    args = parser.parse_args()

    X, y = make_data(args.n, args.p, args.sigma, args.simulation_model, args.seed)

    print(
        f"SIMPLE_TEST n={args.n} p={args.p} sigma={args.sigma} "
        f"model={args.simulation_model} trees={args.trees} depth={args.depth} lr={args.lr}"
    )

    for name, model in build_models(args).items():
        benchmark_one(name, model, X, y)


if __name__ == "__main__":
    main()
