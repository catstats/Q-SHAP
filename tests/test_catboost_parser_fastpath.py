import numpy as np

from qshap import utils


def test_symmetric_parser_processes_each_split_once(monkeypatch):
    depth = 8
    p = 4
    calls = {"feature": 0, "threshold": 0, "default_left": 0}

    original_feature = utils._catboost_split_feature
    original_threshold = utils._catboost_split_threshold
    original_default_left = utils._catboost_split_default_left

    def counted_feature(*args, **kwargs):
        calls["feature"] += 1
        return original_feature(*args, **kwargs)

    def counted_threshold(*args, **kwargs):
        calls["threshold"] += 1
        return original_threshold(*args, **kwargs)

    def counted_default_left(*args, **kwargs):
        calls["default_left"] += 1
        return original_default_left(*args, **kwargs)

    monkeypatch.setattr(utils, "_catboost_split_feature", counted_feature)
    monkeypatch.setattr(utils, "_catboost_split_threshold", counted_threshold)
    monkeypatch.setattr(utils, "_catboost_split_default_left", counted_default_left)

    splits = [
        {
            "split_type": "FloatFeature",
            "float_feature_index": level % p,
            "border": (level + 1) / 10,
        }
        for level in range(depth)
    ]
    num_leaves = 1 << depth
    tree_data = {
        "splits": splits,
        "leaf_values": np.arange(num_leaves, dtype=np.float64) / num_leaves,
        "leaf_weights": np.ones(num_leaves, dtype=np.float64),
    }
    flat_feature_index = {feature: feature for feature in range(p)}
    default_left = {feature: True for feature in range(p)}

    tree = utils.catboost_oblivious_to_simple(
        tree_data,
        flat_feature_index=flat_feature_index,
        default_left_by_feature=default_left,
    )

    assert calls == {
        "feature": depth,
        "threshold": depth,
        "default_left": depth,
    }
    assert tree.node_count == (1 << (depth + 1)) - 1
    for level in range(depth):
        nodes = slice((1 << level) - 1, (1 << (level + 1)) - 1)
        assert np.unique(tree.feature[nodes]).size == 1
        assert np.unique(tree.threshold[nodes]).size == 1
