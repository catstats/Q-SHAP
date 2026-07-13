import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

from qshap._moment import is_read_once_tree, t2_moment
from qshap.qshap import T2, loss_treeshap


def _complex_tables(max_degree):
    store_v_invc = np.zeros((max_degree + 1, max_degree), dtype=complex)
    store_z = np.zeros((max_degree + 1, max_degree), dtype=complex)

    for degree in range(1, max_degree + 1):
        binomial = np.array(
            [math.comb(degree - 1, k) for k in range(degree)], dtype=float
        )
        omega_inv = np.exp(-2 * np.pi * 1j * np.arange(degree) / degree)
        vandermonde = np.vander(omega_inv, increasing=True)
        store_v_invc[degree, :degree] = (
            vandermonde @ (1.0 / binomial) / (degree * degree)
        )
        store_z[degree, :degree] = np.exp(
            2 * np.pi * 1j * np.arange(degree) / degree
        )

    return store_v_invc, store_z


def _coalition_value(x, coalition, tree, node=0):
    left = tree.children_left[node]
    if left < 0:
        return tree.value[node]

    right = tree.children_right[node]
    split_feature = tree.feature[node]
    if split_feature in coalition:
        child = left if x[split_feature] <= tree.threshold[node] else right
        return _coalition_value(x, coalition, tree, child)

    return (
        _coalition_value(x, coalition, tree, left) / tree.sample_weight[left]
        + _coalition_value(x, coalition, tree, right) / tree.sample_weight[right]
    )


def _brute_square_shap(x, tree, n_features):
    result = np.zeros(n_features)
    for feature in range(n_features):
        others = [j for j in range(n_features) if j != feature]
        for size in range(len(others) + 1):
            coefficient = (
                math.factorial(size)
                * math.factorial(n_features - size - 1)
                / math.factorial(n_features)
            )
            for coalition_tuple in itertools.combinations(others, size):
                coalition = set(coalition_tuple)
                without = _coalition_value(x, coalition, tree)
                with_feature = _coalition_value(x, coalition | {feature}, tree)
                result[feature] += coefficient * (
                    with_feature * with_feature - without * without
                )
    return result


def _full_read_once_tree(depth, rng):
    node_count = 2 ** (depth + 1) - 1
    internal_count = 2**depth - 1
    children_left = np.full(node_count, -1, dtype=np.int64)
    children_right = np.full(node_count, -1, dtype=np.int64)
    feature = np.full(node_count, -1, dtype=np.int64)
    threshold = np.zeros(node_count)
    counts = np.zeros(node_count)
    counts[0] = 1000.0

    for node in range(internal_count):
        children_left[node] = 2 * node + 1
        children_right[node] = 2 * node + 2
        feature[node] = node
        threshold[node] = rng.normal()
        fraction = rng.uniform(0.15, 0.85)
        counts[children_left[node]] = counts[node] * fraction
        counts[children_right[node]] = counts[node] * (1.0 - fraction)

    sample_weight = np.ones(node_count)
    for node in range(internal_count):
        sample_weight[children_left[node]] = (
            counts[node] / counts[children_left[node]]
        )
        sample_weight[children_right[node]] = (
            counts[node] / counts[children_right[node]]
        )

    value = np.zeros(node_count)
    value[internal_count:] = rng.normal(size=node_count - internal_count)
    init_prediction = value * counts / counts[0]

    return SimpleNamespace(
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        feature_uniq=np.arange(internal_count, dtype=np.int64),
        threshold=threshold,
        max_depth=depth,
        sample_weight=sample_weight,
        init_prediction=init_prediction,
        value=value,
        node_count=node_count,
    )


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_moment_matches_brute_force(depth):
    rng = np.random.default_rng(100 + depth)
    tree = _full_read_once_tree(depth, rng)
    n_features = 2**depth - 1
    x = rng.normal(size=(3, n_features))

    actual = t2_moment(x, tree)
    expected = np.vstack(
        [_brute_square_shap(row, tree, n_features) for row in x]
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_auto_dispatch_matches_leaf_pair_reference():
    tree = _full_read_once_tree(2, np.random.default_rng(7))
    x = np.array([[0.2, -0.4, 0.8], [-1.0, 0.6, -0.1]])
    store_v_invc, store_z = _complex_tables(2 * tree.max_depth)

    fast = T2(x, tree, store_v_invc, store_z, backend="auto")
    reference = T2(x, tree, store_v_invc, store_z, backend="numba")

    np.testing.assert_allclose(fast, reference, rtol=2e-12, atol=2e-12)


def test_repeated_feature_is_rejected_by_moment_backend():
    tree = _full_read_once_tree(2, np.random.default_rng(2))
    tree.feature[2] = tree.feature[1]
    tree.feature_uniq = np.unique(tree.feature[tree.feature >= 0])

    assert not is_read_once_tree(tree)
    with pytest.raises(ValueError, match="globally read-once"):
        t2_moment(np.zeros((1, 3)), tree)


def test_auto_falls_back_for_repeated_features():
    tree = _full_read_once_tree(2, np.random.default_rng(12))
    tree.feature[2] = tree.feature[1]
    tree.feature_uniq = np.unique(tree.feature[tree.feature >= 0])
    x = np.array([[0.2, -0.4, 0.8], [-1.0, 0.6, -0.1]])
    store_v_invc, store_z = _complex_tables(2 * tree.max_depth)

    automatic = T2(x, tree, store_v_invc, store_z, backend="auto")
    reference = T2(x, tree, store_v_invc, store_z, backend="numba")

    np.testing.assert_allclose(automatic, reference, rtol=2e-12, atol=2e-12)


def test_loss_dispatch_uses_moment_result():
    tree = _full_read_once_tree(2, np.random.default_rng(19))
    x = np.array([[0.2, -0.4, 0.8], [-1.0, 0.6, -0.1]])
    y = np.array([0.7, -1.2])
    t0 = np.array([[0.1, 0.2, -0.3], [-0.4, 0.5, 0.6]])
    learning_rate = 0.35
    store_v_invc, store_z = _complex_tables(2 * tree.max_depth)

    class Explainer:
        def shap_values(self, values):
            np.testing.assert_array_equal(values, x)
            return t0

    actual = loss_treeshap(
        x,
        y,
        tree,
        store_v_invc,
        store_z,
        Explainer(),
        learning_rate=learning_rate,
        backend="auto",
    )
    expected = (
        t2_moment(x, tree) * learning_rate**2
        - 2.0 * (y * (t0 * learning_rate).T).T
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_constant_tree_returns_zero():
    tree = SimpleNamespace(
        children_left=np.array([-1], dtype=np.int64),
        children_right=np.array([-1], dtype=np.int64),
        feature=np.array([-1], dtype=np.int64),
        feature_uniq=np.array([], dtype=np.int64),
        threshold=np.array([0.0]),
        max_depth=0,
        sample_weight=np.array([1.0]),
        init_prediction=np.array([4.0]),
        value=np.array([4.0]),
        node_count=1,
    )

    np.testing.assert_array_equal(
        t2_moment(np.zeros((2, 5)), tree), np.zeros((2, 5))
    )
