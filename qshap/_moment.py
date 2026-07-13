"""Exact second-order TreeSHAP for globally read-once trees.

This module avoids the quadratic leaf-pair expansion used by Q-SHAP.  It is
exact only when every split feature occurs at most once in the whole tree.
For such trees, the feature sets below the two children of every split are
disjoint, so the first two moments of the random-coalition prediction can be
propagated locally.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def is_read_once_tree(summary_tree) -> bool:
    """Return whether every split feature occurs at most once in the tree."""
    children_left = np.asarray(summary_tree.children_left)
    feature = np.asarray(summary_tree.feature)
    split_features = feature[children_left >= 0]
    if split_features.size == 0:
        return True
    if np.any(split_features < 0):
        return False
    return np.unique(split_features).size == split_features.size


def _tree_orders_and_depth(children_left, children_right):
    """Build postorder/preorder traversals and the exact maximum depth."""
    n_nodes = children_left.shape[0]
    if n_nodes == 0:
        raise ValueError("Tree must contain at least one node")

    postorder = []
    preorder = []
    stack = [(0, 0, False)]
    seen = np.zeros(n_nodes, dtype=np.uint8)
    max_depth = 0

    while stack:
        node, depth, expanded = stack.pop()
        if node < 0 or node >= n_nodes:
            raise ValueError("Tree contains an invalid child index")

        if expanded:
            postorder.append(node)
            continue

        if seen[node]:
            raise ValueError("Tree contains a cycle or a node with multiple parents")
        seen[node] = 1
        preorder.append(node)
        max_depth = max(max_depth, depth)
        stack.append((node, depth, True))

        left = int(children_left[node])
        right = int(children_right[node])
        if left < 0:
            if right >= 0:
                raise ValueError("Leaf nodes must have two negative child indices")
        else:
            if right < 0:
                raise ValueError("Internal nodes must have two children")
            # Push right first so left is visited first.
            stack.append((right, depth + 1, False))
            stack.append((left, depth + 1, False))

    if not np.all(seen):
        raise ValueError("Tree contains nodes that are not reachable from the root")

    return (
        np.asarray(postorder, dtype=np.int64),
        np.asarray(preorder, dtype=np.int64),
        max_depth,
    )


def _leaf_values(summary_tree, children_left):
    values = np.asarray(summary_tree.value, dtype=np.float64)
    n_nodes = children_left.shape[0]
    if values.shape[0] != n_nodes:
        raise ValueError("summary_tree.value has the wrong number of nodes")

    values = values.reshape(n_nodes, -1)
    if values.shape[1] != 1:
        raise ValueError("Moment Q-SHAP currently supports scalar regression trees only")

    leaf_values = np.zeros(n_nodes, dtype=np.float64)
    is_leaf = children_left < 0
    leaf_values[is_leaf] = values[is_leaf, 0]
    return leaf_values


@njit
def _t2_moment_core(
    x,
    children_left,
    children_right,
    feature,
    threshold,
    sample_weight,
    leaf_values,
    postorder,
    preorder,
    quad_t,
    quad_w,
):
    n_samples, n_features = x.shape
    n_nodes = children_left.shape[0]
    result = np.zeros((n_samples, n_features), dtype=np.float64)

    # Reused work buffers.  The tree has a single parent per node, but += is
    # used in the reverse pass to keep the adjoint formulas explicit.
    mu = np.zeros(n_nodes, dtype=np.float64)
    nu = np.zeros(n_nodes, dtype=np.float64)
    adj_mu = np.zeros(n_nodes, dtype=np.float64)
    adj_nu = np.zeros(n_nodes, dtype=np.float64)

    for i in range(n_samples):
        for q in range(quad_t.shape[0]):
            t = quad_t[q]
            quadrature_weight = quad_w[q]

            # Bottom-up first- and second-moment pass.
            for order_index in range(postorder.shape[0]):
                node = postorder[order_index]
                left = children_left[node]

                if left < 0:
                    value = leaf_values[node]
                    mu[node] = value
                    nu[node] = value * value
                    continue

                right = children_right[node]
                split_feature = feature[node]
                if x[i, split_feature] <= threshold[node]:
                    hot = left
                    cold = right
                else:
                    hot = right
                    cold = left

                hot_fraction = 1.0 / sample_weight[hot]
                cold_fraction = 1.0 / sample_weight[cold]
                one_minus_t = 1.0 - t

                mean_hot = t + one_minus_t * hot_fraction
                mean_cold = one_minus_t * cold_fraction
                second_hot = t + one_minus_t * hot_fraction * hot_fraction
                second_cold = one_minus_t * cold_fraction * cold_fraction
                cross = 2.0 * one_minus_t * hot_fraction * cold_fraction

                mu[node] = mean_hot * mu[hot] + mean_cold * mu[cold]
                nu[node] = (
                    second_hot * nu[hot]
                    + second_cold * nu[cold]
                    + cross * mu[hot] * mu[cold]
                )

            # Reverse-mode derivative of nu[root] with respect to every node's
            # own inclusion probability t_j, all evaluated at t_j = t.
            for node in range(n_nodes):
                adj_mu[node] = 0.0
                adj_nu[node] = 0.0
            adj_nu[0] = 1.0

            for order_index in range(preorder.shape[0]):
                node = preorder[order_index]
                left = children_left[node]
                if left < 0:
                    continue

                right = children_right[node]
                split_feature = feature[node]
                if x[i, split_feature] <= threshold[node]:
                    hot = left
                    cold = right
                else:
                    hot = right
                    cold = left

                hot_fraction = 1.0 / sample_weight[hot]
                cold_fraction = 1.0 / sample_weight[cold]
                one_minus_t = 1.0 - t

                mean_hot = t + one_minus_t * hot_fraction
                mean_cold = one_minus_t * cold_fraction
                second_hot = t + one_minus_t * hot_fraction * hot_fraction
                second_cold = one_minus_t * cold_fraction * cold_fraction
                cross = 2.0 * one_minus_t * hot_fraction * cold_fraction

                d_mu_dt = (
                    (1.0 - hot_fraction) * mu[hot]
                    - cold_fraction * mu[cold]
                )
                d_nu_dt = (
                    (1.0 - hot_fraction * hot_fraction) * nu[hot]
                    - cold_fraction * cold_fraction * nu[cold]
                    - 2.0 * hot_fraction * cold_fraction * mu[hot] * mu[cold]
                )

                result[i, split_feature] += quadrature_weight * (
                    adj_mu[node] * d_mu_dt + adj_nu[node] * d_nu_dt
                )

                adj_mu[hot] += (
                    adj_mu[node] * mean_hot
                    + adj_nu[node] * cross * mu[cold]
                )
                adj_mu[cold] += (
                    adj_mu[node] * mean_cold
                    + adj_nu[node] * cross * mu[hot]
                )
                adj_nu[hot] += adj_nu[node] * second_hot
                adj_nu[cold] += adj_nu[node] * second_cold

    return result


def t2_moment(x, summary_tree):
    """Compute exact Shapley values of the squared tree game in O(L * D).

    The bound is per explained sample (plus the unavoidable output cost) and
    applies only to globally read-once trees: every split feature must occur at
    most once in the complete tree.  A ``ValueError`` is raised otherwise.
    """
    if not is_read_once_tree(summary_tree):
        raise ValueError(
            "Moment Q-SHAP requires a globally read-once tree: every split "
            "feature must occur at most once in the complete tree."
        )

    x = np.asarray(x, dtype=np.float64, order="C")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim != 2:
        raise ValueError("x must be a one- or two-dimensional array")

    children_left = np.asarray(summary_tree.children_left, dtype=np.int64)
    children_right = np.asarray(summary_tree.children_right, dtype=np.int64)
    feature = np.asarray(summary_tree.feature, dtype=np.int64)
    threshold = np.asarray(summary_tree.threshold, dtype=np.float64)
    sample_weight = np.asarray(summary_tree.sample_weight, dtype=np.float64)

    n_nodes = children_left.shape[0]
    if not (
        children_right.shape[0]
        == feature.shape[0]
        == threshold.shape[0]
        == sample_weight.shape[0]
        == n_nodes
    ):
        raise ValueError("Tree arrays must all have the same number of nodes")

    postorder, preorder, max_depth = _tree_orders_and_depth(
        children_left, children_right
    )
    if max_depth == 0:
        return np.zeros_like(x)

    internal = children_left >= 0
    if np.any(feature[internal] >= x.shape[1]):
        raise ValueError("x has fewer columns than the tree's largest feature index")

    non_root = np.arange(n_nodes) != 0
    if np.any(~np.isfinite(sample_weight[non_root])) or np.any(
        sample_weight[non_root] <= 0
    ):
        raise ValueError("Tree sample weights must be finite and positive")

    leaf_values = _leaf_values(summary_tree, children_left)

    # The diagonal Shapley integrand has degree at most 2D - 1.  D-point
    # Gauss-Legendre quadrature therefore evaluates its integral exactly.
    roots, weights = np.polynomial.legendre.leggauss(max_depth)
    quad_t = np.ascontiguousarray((roots + 1.0) * 0.5, dtype=np.float64)
    quad_w = np.ascontiguousarray(weights * 0.5, dtype=np.float64)

    return _t2_moment_core(
        x,
        children_left,
        children_right,
        feature,
        threshold,
        sample_weight,
        leaf_values,
        postorder,
        preorder,
        quad_t,
        quad_w,
    )
