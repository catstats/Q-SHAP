"""Complex-root general-tree quadratic Q-SHAP development hooks.

The public generic backend remains the correctness baseline. This module
provides a stable import location for the experimental O(L^2 D) unordered
leaf-pair kernel while validation against the legacy stable T2 implementation
is expanded.

All polynomial evaluation must use the Q-SHAP complex root-of-unity basis
(`store_z` / `store_v_invc`).
"""

from __future__ import annotations

from qshap.qshap import T2


def t2_ol2d(x, summary_tree, store_v_invc, store_z, *, experimental=False):
    """Experimental T2 interface using the existing stable Q-SHAP basis.

    The depth-linear pair update is not the default until fully validated.
    This function currently delegates to T2, which uses store_z/store_v_invc.
    """
    if experimental:
        # Keep the interface explicit without silently changing numerics.
        pass
    return T2(x, summary_tree, store_v_invc, store_z, backend="auto")


t2_ol2d_baseline = t2_ol2d
