from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from qshap import gazer


def test_get_tree_returns_stored_tree():
    X, y = make_regression(n_samples=20, n_features=2, random_state=42)
    model = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y)
    explainer = gazer(model)

    tree = explainer.get_tree()

    assert tuple(tree) == (
        "children_left", "children_right", "feature", "threshold",
        "max_depth", "n_node_samples", "value", "node_count",
        "default_left", "xgboost_split",
    )
    assert tree["node_count"] == model.tree_.node_count
