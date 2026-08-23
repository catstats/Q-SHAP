#!/usr/bin/env python3
"""Single-run Python counterpart of the CatBoost Q-SHAP time study."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qshap import gazer
from qshap.boosting_importance import fast_catboost_qshap_r2


DEFAULT_NS = (1_000, 5_000, 10_000, 50_000, 100_000)


def make_catboost_growth_data(
    n: int,
    p: int,
    sigma: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the same linear-Gaussian design used by time_study.R."""
    if p < 1:
        raise ValueError("p must be positive")

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.asarray([4.0, -5.0, 6.0, 3.0, -1.0])[: min(5, p)]
    y = X[:, : beta.size] @ beta + rng.normal(scale=sigma, size=n)
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


def run_one(
    n: int,
    *,
    p: int,
    iterations: int,
    depth: int | None,
    learning_rate: float | None,
    sigma: float,
    seed: int,
    thread_count: int,
) -> dict[str, float | int]:
    print(
        f"[n={n}] generating data: p={p}, "
        f"dense X approx {n * p * 8 / 1024**3:.2f} GB",
        flush=True,
    )
    # time_study.R used repeat_id=1 for the first replicate.  This no-repeat
    # version fixes that same offset for every sample-size setting.
    X, y = make_catboost_growth_data(
        n=n,
        p=p,
        sigma=sigma,
        seed=seed + n + 1_000_003,
    )
    pool = Pool(X, label=y)

    params: dict[str, object] = {
        "loss_function": "RMSE",
        "iterations": iterations,
        "random_seed": seed + 1,
        "thread_count": thread_count,
        "verbose": False,
        "allow_writing_files": False,
    }
    if depth is not None:
        params["depth"] = depth
    if learning_rate is not None:
        params["learning_rate"] = learning_rate

    print(f"[n={n}] fitting CatBoost...", flush=True)
    start = time.perf_counter()
    model = CatBoostRegressor(**params).fit(pool)
    fit_sec = time.perf_counter() - start

    print(f"[n={n}] predicting...", flush=True)
    start = time.perf_counter()
    model.predict(pool, thread_count=thread_count)
    predict_sec = time.perf_counter() - start

    print(f"[n={n}] constructing gazer explainer...", flush=True)
    start = time.perf_counter()
    explainer = gazer(model)
    gazer_sec = time.perf_counter() - start

    print(f"[n={n}] computing Q-SHAP...", flush=True)
    start = time.perf_counter()
    rsq = fast_catboost_qshap_r2(
        explainer,
        X,
        y,
        compute_sd=False,
    )
    qshap_rsq_sec = time.perf_counter() - start
    qshap_sec = gazer_sec + qshap_rsq_sec

    print(
        f"[n={n}] done: fit={fit_sec:.3f}s, "
        f"predict={predict_sec:.3f}s, gazer={gazer_sec:.3f}s, "
        f"Q-SHAP core={qshap_rsq_sec:.3f}s, total={qshap_sec:.3f}s",
        flush=True,
    )

    return {
        "n": n,
        "p": p,
        "iterations": iterations,
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "gazer_sec": gazer_sec,
        "qshap_rsq_sec": qshap_rsq_sec,
        "qshap_sec": qshap_sec,
        "rsq_checksum": float(np.sum(rsq)),
    }


def write_results(
    rows: list[dict[str, float | int]],
    out_dir: Path,
    out_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{out_prefix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    n = np.asarray([row["n"] for row in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        n,
        [max(float(row["fit_sec"]), 1e-6) for row in rows],
        color="#D55E00",
        marker="o",
        label="CatBoost training",
    )
    ax.plot(
        n,
        [max(float(row["predict_sec"]), 1e-6) for row in rows],
        color="#7F7F7F",
        linestyle="--",
        marker="o",
        label="CatBoost prediction",
    )
    ax.plot(
        n,
        [max(float(row["qshap_sec"]), 1e-6) for row in rows],
        color="#0072B2",
        marker="o",
        label="Q-SHAP",
    )
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Run time (s)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    pdf_path = out_dir / f"{out_prefix}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote PDF: {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", nargs="+", type=int, default=list(DEFAULT_NS))
    parser.add_argument("--p", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("qshap_catboost_growth_artifacts_python"),
    )
    parser.add_argument(
        "--out-prefix",
        default="catboost_qshap_sample_growth_python",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(n < 1 for n in args.ns):
        raise ValueError("all sample sizes must be positive")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.thread_count < 1:
        raise ValueError("thread-count must be positive")

    rows = [
        run_one(
            n,
            p=args.p,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            sigma=args.sigma,
            seed=args.seed,
            thread_count=args.thread_count,
        )
        for n in args.ns
    ]
    write_results(rows, args.out_dir, args.out_prefix)


if __name__ == "__main__":
    main()
