"""Private CatBoost routing and symmetric-tree helpers for Q-SHAP."""

from __future__ import annotations

from math import comb

import numpy as np


def catboost_float_features(X):
    """Match CatBoost's float32 feature quantization before tree routing."""
    return np.asarray(X, dtype=np.float32, order="C").astype(
        np.float64, copy=False
    )


def _level_splits(tree):
    depth = int(tree.max_depth)
    features = np.empty(depth, dtype=np.int64)
    thresholds = np.empty(depth, dtype=np.float64)
    for level in range(depth):
        node = (1 << level) - 1
        features[level] = int(tree.feature[node])
        thresholds[level] = float(tree.threshold[node])
    return features, thresholds


def route_oblivious_leaves(X, tree):
    """Route all rows to 0-based CatBoost leaf ids."""
    X = catboost_float_features(X)
    features, thresholds = _level_splits(tree)
    leaf = np.zeros(X.shape[0], dtype=np.int64)
    for level, (feature, threshold) in enumerate(zip(features, thresholds)):
        node = (1 << level) - 1
        values = X[:, feature]
        default_left = (
            True if tree.default_left is None else bool(tree.default_left[node])
        )
        go_right = np.where(
            np.isnan(values),
            not default_left,
            values > threshold,
        )
        leaf = leaf * 2 + go_right.astype(np.int64)
    return leaf


def route_tree_leaf_nodes(X, tree):
    """Route rows through any parsed numeric CatBoost tree.

    Returns node indices, rather than compact leaf ordinals, because general
    Depthwise and Lossguide trees are not complete binary trees.
    """
    X = catboost_float_features(X)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")

    nodes = np.zeros(X.shape[0], dtype=np.int64)
    active = tree.children_left[nodes] >= 0
    while np.any(active):
        for node in np.unique(nodes[active]):
            rows = active & (nodes == node)
            feature = int(tree.feature[node])
            values = X[rows, feature]
            default_left = (
                True if tree.default_left is None else bool(tree.default_left[node])
            )
            go_left = np.where(
                np.isnan(values),
                default_left,
                values <= float(tree.threshold[node]),
            )
            nodes[rows] = np.where(
                go_left,
                int(tree.children_left[node]),
                int(tree.children_right[node]),
            )
        active = tree.children_left[nodes] >= 0
    return nodes


def _suffix_features(level_features):
    depth = len(level_features)
    suffix = [None] * (depth + 1)
    suffix[depth] = []
    for level in range(depth - 1, -1, -1):
        current = list(suffix[level + 1])
        feature = int(level_features[level])
        if feature not in current:
            current.append(feature)
        suffix[level] = current
    return suffix


def _project_masks(curr_features, next_features):
    curr_pos = {feature: idx for idx, feature in enumerate(curr_features)}
    projection = np.zeros(1 << len(curr_features), dtype=np.int64)
    for mask in range(projection.shape[0]):
        out = 0
        for next_pos, feature in enumerate(next_features):
            curr_idx = curr_pos.get(feature)
            if curr_idx is not None and (mask & (1 << curr_idx)):
                out |= 1 << next_pos
        projection[mask] = out
    return projection


def _path_bits(leaf_id, depth):
    return [int((leaf_id >> (depth - 1 - level)) & 1) for level in range(depth)]


def _shapley_from_subset_values_compact(values, features, n_features):
    feature_ids = []
    T0_by_position = np.zeros(len(features), dtype=np.float64)
    T2_by_position = np.zeros(len(features), dtype=np.float64)
    k = len(features)
    if k == 0:
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)

    squared = values * values
    for mask in range(1 << k):
        if mask == (1 << k) - 1:
            continue
        subset_size = int(mask.bit_count())
        weight = 1.0 / (k * comb(k - 1, subset_size))
        for bit, feature in enumerate(features):
            if mask & (1 << bit):
                continue
            with_feature = mask | (1 << bit)
            T0_by_position[bit] += weight * (values[with_feature] - values[mask])
            T2_by_position[bit] += weight * (squared[with_feature] - squared[mask])

    T0_values = []
    T2_values = []
    for bit, feature in enumerate(features):
        if 0 <= feature < n_features and (
            T0_by_position[bit] != 0.0 or T2_by_position[bit] != 0.0
        ):
            feature_ids.append(feature)
            T0_values.append(T0_by_position[bit])
            T2_values.append(T2_by_position[bit])

    return (
        np.asarray(feature_ids, dtype=np.int64),
        np.asarray(T0_values, dtype=np.float64),
        np.asarray(T2_values, dtype=np.float64),
    )


def _shapley_from_subset_values(values, features, n_features):
    feature_ids, T0_values, T2_values = _shapley_from_subset_values_compact(
        values, features, n_features
    )
    T0_leaf = np.zeros(n_features, dtype=np.float64)
    T2_leaf = np.zeros(n_features, dtype=np.float64)
    T0_leaf[feature_ids] = T0_values
    T2_leaf[feature_ids] = T2_values
    return T0_leaf, T2_leaf


def catboost_leaf_T0_T2_compact(tree, leaf_id, n_features):
    """Exact nonzero T0/T2 terms for one reached CatBoost leaf/path."""
    depth = int(tree.max_depth)
    if depth == 0:
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)

    level_features, _ = _level_splits(tree)
    suffix = _suffix_features(level_features)
    hot_bits = _path_bits(int(leaf_id), depth)

    num_leaves = 1 << depth
    num_internal = num_leaves - 1
    dp_next = [
        np.array([float(tree.value[num_internal + leaf])], dtype=np.float64)
        for leaf in range(num_leaves)
    ]

    for level in range(depth - 1, -1, -1):
        curr_features = suffix[level]
        next_features = suffix[level + 1]
        projection = _project_masks(curr_features, next_features)
        split_pos = curr_features.index(int(level_features[level]))
        nodes_this_level = 1 << level
        dp_curr = []

        for node_pos in range(nodes_this_level):
            bfs_node = (1 << level) - 1 + node_pos
            left_bfs = int(tree.children_left[bfs_node])
            right_bfs = int(tree.children_right[bfs_node])
            left_pos = 2 * node_pos
            right_pos = 2 * node_pos + 1
            parent_n = max(float(tree.n_node_samples[bfs_node]), np.finfo(float).tiny)
            p_left = float(tree.n_node_samples[left_bfs]) / parent_n
            p_right = float(tree.n_node_samples[right_bfs]) / parent_n

            values = np.empty(1 << len(curr_features), dtype=np.float64)
            for mask in range(values.shape[0]):
                next_mask = int(projection[mask])
                if mask & (1 << split_pos):
                    values[mask] = (
                        dp_next[right_pos][next_mask]
                        if hot_bits[level]
                        else dp_next[left_pos][next_mask]
                    )
                else:
                    values[mask] = (
                        p_left * dp_next[left_pos][next_mask]
                        + p_right * dp_next[right_pos][next_mask]
                    )
            dp_curr.append(values)

        dp_next = dp_curr

    return _shapley_from_subset_values_compact(dp_next[0], suffix[0], n_features)


def catboost_leaf_T0_T2(tree, leaf_id, n_features):
    """Exact dense T0 and T2 Q-SHAP terms for one reached CatBoost leaf/path."""
    feature_ids, T0_values, T2_values = catboost_leaf_T0_T2_compact(
        tree, leaf_id, n_features
    )
    T0_leaf = np.zeros(n_features, dtype=np.float64)
    T2_leaf = np.zeros(n_features, dtype=np.float64)
    T0_leaf[feature_ids] = T0_values
    T2_leaf[feature_ids] = T2_values
    return T0_leaf, T2_leaf
