"""Compatibility wrapper for the complex-root general-tree T2 hook.

This legacy filename is kept so old imports do not break. The implementation
must stay aligned with the Q-SHAP stable polynomial basis: complex roots of
unity through ``store_z`` / ``store_v_invc``.
"""

from __future__ import annotations

from qshap.product_complex_t2 import t2_ol2d, t2_ol2d_baseline

__all__ = ["t2_ol2d", "t2_ol2d_baseline"]
