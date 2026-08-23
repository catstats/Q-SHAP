"""Cached global R_j^2 backend for CatBoost symmetric trees."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from qshap._backend import catboost_qshap_r2_fast_cpp, cpp_available
from qshap.catboost_backend import (
    catboost_float_features,
    catboost_leaf_T0_T2_compact,
    route_oblivious_leaves,
)


def catboost_global_loss_stats(explainer, X, y, compute_sd=True):
    """Return sufficient statistics for global CatBoost Q-SHAP R2.

    This backend is for global R_j^2 only. If callers need local loss matrices,
    the generic backend remains the correct path.

    CatBoost JSON leaf values are already scaled by the model learning rate.
    """
    if cpp_available():
        stats = catboost_qshap_r2_fast_cpp(
            X, y, explainer.catboost_res, explainer.base_score, compute_sd=compute_sd
        )
        return SimpleNamespace(
            rsq=stats["rsq"],
            loss_sum=stats["loss_sum"],
            loss_sumsq=stats["loss_sumsq"],
            sd_rsq=stats["sd_rsq"],
            n=int(stats["n"]),
            sst=float(stats["sst"]),
        )

    return _catboost_global_loss_stats_python(explainer, X, y, compute_sd=compute_sd)


def _catboost_global_loss_stats_python(explainer, X, y, compute_sd=True):
    """Pure Python fallback used only when the compiled extension is unavailable."""
    X = catboost_float_features(X)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, p = X.shape

    cumulative_prediction = np.full(n, explainer.base_score, dtype=np.float64)
    loss_sum = np.zeros(p, dtype=np.float64)
    total_loss = np.zeros((n, p), dtype=np.float64) if compute_sd else None

    for tree in explainer.catboost_res:
        residual = y - cumulative_prediction
        leaf_ids = route_oblivious_leaves(X, tree)
        num_leaves = 1 << int(tree.max_depth)
        num_internal = num_leaves - 1
        if int(tree.node_count) != 2 * num_leaves - 1:
            raise NotImplementedError(
                "Fast CatBoost Q-SHAP currently supports regression with symmetric numeric trees."
            )

        for leaf_id in np.unique(leaf_ids):
            group = leaf_ids == leaf_id
            count = int(np.sum(group))
            if count == 0:
                continue

            group_residual = residual[group]
            sum_r = float(np.sum(group_residual))
            feature_ids, T0_leaf, T2_leaf = catboost_leaf_T0_T2_compact(
                tree, int(leaf_id), p
            )

            for idx, j in enumerate(feature_ids):
                a = float(T2_leaf[idx])
                b = 2.0 * float(T0_leaf[idx])
                loss_sum[j] += count * a - b * sum_r
                if compute_sd:
                    total_loss[group, j] += a - b * group_residual

        cumulative_prediction += tree.value[num_internal + leaf_ids]

    loss_sumsq = (
        np.sum(total_loss * total_loss, axis=0) if compute_sd else None
    )

    return SimpleNamespace(
        loss_sum=loss_sum,
        loss_sumsq=loss_sumsq,
        n=n,
    )
