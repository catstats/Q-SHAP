"""Global boosting-model Q-SHAP importance helpers."""

from __future__ import annotations

import numpy as np

from qshap.catboost_cached_importance import catboost_global_loss_stats


def _as_catboost_explainer(model):
    if getattr(model, "model_kind", None) == "catboost" and hasattr(model, "catboost_res"):
        return model
    try:
        import catboost
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise NotImplementedError(
            "Fast CatBoost Q-SHAP currently supports regression with symmetric numeric trees."
        ) from exc
    if isinstance(model, catboost.CatBoostRegressor):
        from qshap.main import gazer

        return gazer(model)
    raise NotImplementedError(
        "Fast CatBoost Q-SHAP currently supports regression with symmetric numeric trees."
    )


def fast_catboost_qshap_r2(model, X, y, compute_sd=True):
    """Compute global CatBoost feature-specific R_j^2 without local loss matrix."""
    explainer = _as_catboost_explainer(model)
    if not getattr(explainer, "catboost_is_symmetric", False):
        raise NotImplementedError(
            "The fast CatBoost backend requires grow_policy='SymmetricTree'. "
            "Use gazer(model).rsq(...) for Depthwise or Lossguide models."
        )
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    stats = catboost_global_loss_stats(explainer, X, y, compute_sd=compute_sd)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 0:
        raise ValueError("Cannot compute R2 decomposition when y has zero variance")
    rsq = -stats.loss_sum / sst
    if not compute_sd:
        return rsq

    if stats.n > 1:
        loss_var = np.maximum(
            (stats.loss_sumsq - (stats.loss_sum * stats.loss_sum) / stats.n) / (stats.n - 1),
            0.0,
        )
        sd_rsq = np.sqrt(stats.n * loss_var) / sst
    else:
        sd_rsq = np.full_like(rsq, np.nan)

    from types import SimpleNamespace

    return SimpleNamespace(
        rsq=rsq,
        sd_rsq=sd_rsq,
        loss_sum=stats.loss_sum,
        loss_sumsq=stats.loss_sumsq,
        n=stats.n,
        sst=sst,
    )


catboost_global_rsq = fast_catboost_qshap_r2
