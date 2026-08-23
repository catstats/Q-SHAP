import numpy as np
from numba import njit
import time as time
from dataclasses import dataclass
import json
#d_test = 9

def inv_binom_coef(d):
    """
    calculate the inverse of binomial coefficients using an iterative method with symmetry.
    
    Parameters:
    - d: dimension
    
    Example: 
    binom_coef(5)
    ([ 1.,  5., 10., 10.,  5.,  1.])
    """
    coef = np.zeros(d + 1)
    coef[0] = 1
    for i in range(1, d // 2 + 1):
        coef[i] = coef[i - 1] * (d - i + 1) / i
    for i in range(d // 2 + 1, d + 1):
        coef[i] = coef[d - i]
    return 1/coef

#inv_binom_coef(d_test)

def complex_v_invc_degree(d):
    """
    Pre store v_invc: v(z)^-1 @ c / d at degree d where z are complex roots of unity
    
    Parameters:
    -d: degree
    -c: the coefficients matrix
    """
    omega_inv = np.exp(-2 * np.pi * 1j * np.arange(d) / d)
    v_omega_inv = np.vander(omega_inv, increasing=True)
    v_inv_omega_theo = v_omega_inv / d
    res = v_inv_omega_theo @ inv_binom_coef(d-1) / d
    return res

#print(complex_v_invc_degree(d_test))

def store_complex_v_invc(d):
    """
    Pre store v_invc: v(z)^-1 @ c / d up to maximum tree depth where z are complex roots of unity
    
    Parameters:
    -d: max treedepth
    """
    res = np.zeros((d+1, d), dtype=complex)
    
    for i in range(1, d+1):
        res[i, :i] = complex_v_invc_degree(i)
    
    return res 

# how to retrive the degree d_test pre_stored value
# print(store_complex_v_invc(20)[d_test, :d_test])

def store_complex_root(d):
    """
    Prestore the complex root of unity z
    
    Parameters:
    -d: max treedepth
    """ 
    res = np.zeros((d+1, d), dtype=complex)
    
    for i in range(1, d+1):
        res[i, :i] = np.exp(2 * np.pi * 1j * np.arange(i) / i)
    return res


@njit
def complex_dot_v2(p, v_invc, d):
    """
    Return the dot product: C(z) * P(z) / d where z are the complex roots of unity, using the fact that P(w) and v_invc both:
    except for the 0 index and possibly the last when d is odd, the rest are complex conjugate by head and tail, etc...
    
    Parameters 
    - p: a polynomial vector evaluated at complex root of unity
    - v_invc: pre-calculated inverse coefficients
    - d: the original degree before cut by half.
    """
    len_p = len(p)
    res = p[0] * v_invc[0]
    if d % 2 == 0:
        res += 2 * np.dot(p[1:(len_p-1)], v_invc[1:(len_p-1)]) + p[-1] * v_invc[-1]
    else:
        res += 2 * np.dot(p[1:len_p], v_invc[1:len_p])
    return res.real

# v1 = np.array([3, 6-1j, 5+1j, 6+1j])
# v2 = np.array([5, 4-1j, 6+1j, 4+1j])
# v3 = np.array([3, 6-1j, 5+1j])
# v4 = np.array([5, 4-1j, 6+1j])
# print(complex_dot_v2(v3, v4, len(v1)))

# v5 = np.array([3, 6-1j, 5+1j, 5-1j, 6+1j])
# v6 = np.array([5, 4-1j, 6+1j, 6-1j, 4+1j])
# v7 = np.array([3, 6-1j, 5+1j])
# v8 = np.array([5, 4-1j, 6+1j])
# print(complex_dot_v2(v7, v8, len(v5)))


@dataclass(frozen=True)
class simple_tree:
    """
    dataclass for a simple tree, in the scikit learn format that is necessary for computation
    
    Data:
    children_left: left children index
    children_right: right children index
    feature: array of features splitted at each node
    threshold: array of thresholds for corresponding splitting features
    max_depth: max_depth of the tree
    n_node_samples: array of sample size for each node
    value: array of values for each node, only leaf value is used, so only keep leaf value is fine
    node_count: total number of leaves 
    """
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    max_depth: int
    n_node_samples: np.ndarray
    value: np.ndarray
    node_count: int
    default_left: object = None
    xgboost_split: bool = False


@dataclass(frozen=True)
class tree_summary:
    """
    dataclass for the calculation of cd-treeshap family
    
    Data:
    - children_left: left children_index
    - children_right: right children_index 
    - feature: array of features splitted at each node
    - feature_uniq: array of uniq features
    - threshold: array of threshols for corresponding splitting features
    - max_depth: the max_depth of the tree
    - sample_weight: list of sample size of parent/sample size of current node
    - init_prediction: initial prediction from each leaf
    - value: array of values
    - n_node_samples: array of node counts
    - node_count: number of nodes
    """
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    feature_uniq: np.ndarray
    threshold: np.ndarray
    max_depth: int
    sample_weight: np.ndarray
    init_prediction: np.ndarray
    value: np.ndarray
    n_node_samples: np.ndarray
    node_count: int
    default_left: object = None
    xgboost_split: bool = False
    

def _as_single_output_tree_values(value):
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 1:
        return values

    squeezed = np.squeeze(values)
    if squeezed.ndim == 1 and squeezed.shape[0] == values.shape[0]:
        return np.ascontiguousarray(squeezed, dtype=np.float64)

    raise ValueError("Only single-output regression trees are supported")


    
def summarize_tree(tree):
    """
    Summarize the data needed for tree_summary. The tree object should have:
    children_left: left children_index
    children_right: right children_index
    feature: array of features splitted at each node
    threshold: array of threshols for corresponding splitting features
    max_depth: max_depth of the tree
    value: array of values for each node, only leaf value is used, so only keep leaf value is fine
    n_node_samples: array of sample size for each node
    node_count: total number of leaves 
    """
    sample_weight = np.ones_like(tree.threshold)
    init_prediction = np.zeros_like(tree.threshold)
    tree_value = _as_single_output_tree_values(tree.value)
    default_left = getattr(tree, "default_left", None)
    if default_left is None:
        default_left = np.zeros(tree.node_count, dtype=np.bool_)
    else:
        default_left = np.asarray(default_left, dtype=np.bool_)
    xgboost_split = bool(getattr(tree, "xgboost_split", False))
    n = tree.n_node_samples[0]
    
    def traversal_summarize_tree(v):
        v_l, v_r = tree.children_left[v], tree.children_right[v]
        n_v = tree.n_node_samples[v]

        init_prediction[v] = tree_value[v] * n_v/n
        
        if v_l < 0:  #leaf
            return
        else:
            n_l, n_r = tree.n_node_samples[v_l], tree.n_node_samples[v_r]
            sample_weight[v_l], sample_weight[v_r] = n_v/n_l, n_v/n_r
            traversal_summarize_tree(v_l)
            traversal_summarize_tree(v_r)
    
    # travel from the root
    traversal_summarize_tree(0)
    
    feature_uniq = np.unique(tree.feature[tree.feature >= 0])
   
    return tree_summary(
        tree.children_left, tree.children_right, tree.feature, feature_uniq,
        tree.threshold, tree.max_depth, sample_weight, init_prediction,
        tree_value, tree.n_node_samples, tree.node_count, default_left,
        xgboost_split,
    )


def traversal_weight(x, v, w, children_left, children_right, feature, threshold,
                     default_left, xgboost_split, sample_weight, leaf_ind,
                     w_res, w_ind, depth, met_feature):
    """
    Calculate the weight in the treeSHAP. 

    Parameters:
    - x: one sample to be explained 
    - v: node index 
    - w: weight vector passed to the current node, for temporary usage
    - children_left: left children_index
    - children_right: right children_index 
    - feature: list of features splitted at each node
    - threshold: list of threshold for corresponding features
    - sample_weight: a list of sample size of parent/sample size of current node
    - leaf_ind: leaf indices
    - w_res, L * p matrix of weights, which records the modified weight for each leaf and each feature
    - w_ind, L * p matrix of indicator matrix, which records the met of features for each leaf
    - depth: current depth
    - met_feature: record all the features met to now

    Update:
    w_res
    w_ind
    met_feature
    """

    v_l, v_r = children_left[v], children_right[v]

    if v_l < 0:
        # match to the right location so the value for w_res corresponds to the same order of leaf_ind
        ind = (leaf_ind == v)
        #feature_tmp = met_feature[0:depth]
        for tmp_depth in range(depth):
            w_res[ind, met_feature[tmp_depth]] = w[tmp_depth]
            w_ind[ind, met_feature[tmp_depth]] = 1
    else:
        split_feature = feature[v]
        split_threshold = threshold[v]

        former_depth = np.arange(depth)[met_feature[0:depth] == split_feature]

        if len(former_depth) != 0:
            former_depth = former_depth[-1]
        else:
            former_depth = depth  
            w[depth] = 1

        met_feature = met_feature.copy()
        met_feature[depth] = split_feature

        w_r = w.copy()

        split_value = x[split_feature]
        if np.isnan(split_value):
            go_left = default_left[v]
        elif xgboost_split:
            go_left = np.float32(split_value) < np.float32(split_threshold)
        else:
            go_left = split_value <= split_threshold

        if go_left:
            w[depth] = w[former_depth] * sample_weight[v_l]
            w_r[depth] = 0
        else:
            w_r[depth] = w_r[former_depth] * sample_weight[v_r]
            w[depth] = 0 

        traversal_weight(x, v_l, w, children_left, children_right, feature,
                         threshold, default_left, xgboost_split, sample_weight,
                         leaf_ind, w_res, w_ind, depth+1, met_feature)
        traversal_weight(x, v_r, w_r, children_left, children_right, feature,
                         threshold, default_left, xgboost_split, sample_weight,
                         leaf_ind, w_res, w_ind, depth+1, met_feature)

        
def weight(x, summary_tree):
    p = len(x)
    d = summary_tree.max_depth
    
    feature_uniq = summary_tree.feature_uniq
    
    leaf_ind = np.arange(summary_tree.node_count)[summary_tree.children_left==-1]

    # L * p matrix. Note that unused features are also labeled as 0 here for efficient storage.
    w_res = np.empty((len(leaf_ind), p))
    w_res[:, feature_uniq] = 1
    
    w = np.empty(d)
    
    # L * p matrix. [i, j] = 1 if the feature j is used by leaf i. [i, j] = 0 corresponds to 1 in the
    # above matrix. This two sparse matrices together could make the weight matrix well-defined, and save the
    # storage at the same time
    w_ind = np.empty((len(leaf_ind), p))
    w_ind[:, feature_uniq] = 0
    
    met_feature = np.full(d, -1, dtype=int)
    
    # begin traversal from root
    traversal_weight(
        x, 0, w, summary_tree.children_left, summary_tree.children_right,
        summary_tree.feature, summary_tree.threshold, summary_tree.default_left,
        summary_tree.xgboost_split, summary_tree.sample_weight, leaf_ind,
        w_res, w_ind, 0, met_feature,
    )
    return w_res, w_ind


def xgb_formatter(model_data, max_depth):
    """
    This function takes the json format of the xgboost output and transform it to a list that treeshap rsq can understand
    
    Parameters:
    model_data: json file
    max_depth: the max tree depth
    
    Examples:
    import json
    xgb_regressor.save_model("model.json")
    with open('model.json', 'r') as file:
    model_data = json.load(file)
    xgb_tree_res = xgb_formatter(model_data, 4)
    """
    trees_data = model_data["learner"]["gradient_booster"]["model"]["trees"]

    xgb_tree = []

    for tree in trees_data:
        # XGBoost stores and evaluates numeric features as float32, routes
        # equality to the right, and learns a default direction for missing
        # values. Keep the raw float32 thresholds and routing flags so the
        # shared traversal can reproduce those decisions exactly.
        threshold = np.asarray(
            tree["split_conditions"], dtype=np.float32
        ).astype(np.float64)
        default_left = np.asarray(
            tree.get("default_left", np.zeros(len(threshold))), dtype=np.bool_
        )

        xgb_tree.append(simple_tree(np.array(tree["left_children"]),
                                    np.array(tree["right_children"]), 
                                    np.array(tree["split_indices"]),
                                    threshold,
                                    max_depth, 
                                    np.array(tree["sum_hessian"]), 
                                    np.array(tree["base_weights"]),
                                    int(tree["tree_param"]["num_nodes"]),
                                    default_left,
                                    True))

    return(xgb_tree)


def lgb_formatter(model_data, max_depth):
    """
    This function takes the trees_to_dataframe() format of the LightGBM output and transform it to a list that treeshap rsq can understand
    
    Parameters:
    model_data: the output of trees_to_dataframe() file
    max_depth: the max tree depth
    
    Examples:
    lgb_tree_res = lgb_formatter(model_data, max_depth)
    """
    ntree = model_data['tree_index'].iloc[-1] + 1

    lgb_tree = []

    def _is_missing(value):
        return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))

    def _parse_split_feature(value):
        if _is_missing(value):
            return -1
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return -1 if np.isnan(value) else int(value)
        value = str(value)
        if value.startswith("Column_"):
            return int(value.replace("Column_", ""))
        if value.startswith("feature_"):
            return int(value.replace("feature_", ""))
        return int(value)

    def _map_child(value, node_mapping):
        if _is_missing(value):
            return -1
        return node_mapping[value]

    for tree_id in range(ntree):
        
        tree = model_data[model_data['tree_index']==tree_id] 

        node_mapping = {original: idx for idx, original in enumerate(tree['node_index'])}
        node_mapping[None] = - 1

        split_feature = np.array(
            [_parse_split_feature(f) for f in tree['split_feature']],
            dtype=np.int64,
        )
        
        lgb_tree.append(simple_tree(np.array([_map_child(v, node_mapping) for v in tree['left_child']], dtype=np.int64),
                                    np.array([_map_child(v, node_mapping) for v in tree['right_child']], dtype=np.int64),
                                    split_feature,
                                    np.nan_to_num(np.array(tree["threshold"], dtype=np.float64), nan=0.0),
                                    max_depth, 
                                    np.array(tree["count"], dtype=np.float64),
                                    np.array(tree["value"], dtype=np.float64),
                                    int(tree.shape[0]),
                                    np.array(tree["missing_direction"] == "left", dtype=np.bool_)))
    return(lgb_tree)


def _catboost_float_feature_metadata(model_data):
    """Return CatBoost float-feature index mappings and missing directions."""
    flat_feature_index = {}
    default_left = {}
    features_info = model_data.get("features_info")
    if not isinstance(features_info, dict) or "float_features" not in features_info:
        return None, None
    float_features = features_info.get("float_features") or []

    for position, feature_info in enumerate(float_features):
        feature_index = int(feature_info.get("feature_index", position))
        flat_feature_index[feature_index] = int(
            feature_info.get("flat_feature_index", feature_index)
        )
        # CatBoost evaluates FloatFeature splits as ``value > border``.
        # AsFalse (nan_mode=Min) therefore routes missing values left, while
        # AsTrue (nan_mode=Max) routes them right.  AsIs is emitted when the
        # training feature had no missing values and follows the ordinary
        # false/left result of a NaN comparison.
        default_left[feature_index] = (
            feature_info.get("nan_value_treatment", "AsIs") != "AsTrue"
        )
    return flat_feature_index, default_left


def _catboost_split_feature(split_info, flat_feature_index=None):
    split_type = split_info.get("split_type", "FloatFeature")
    if split_type != "FloatFeature":
        raise NotImplementedError(
            "CatBoost support currently handles numeric FloatFeature splits only. "
            "Train CatBoost on numeric features for qshap, or add categorical "
            "split handling before calling gazer()."
        )

    if "float_feature_index" in split_info:
        feature_index = int(split_info["float_feature_index"])
        if flat_feature_index is not None:
            if feature_index not in flat_feature_index:
                raise ValueError(
                    "CatBoost split references float_feature_index "
                    f"{feature_index}, but features_info has no matching feature"
                )
            return int(flat_feature_index[feature_index])
        return feature_index
    if "flat_feature_index" in split_info:
        return int(split_info["flat_feature_index"])
    raise ValueError("Cannot find float feature index in CatBoost split info")


def _catboost_split_threshold(split_info):
    if "border" in split_info:
        return float(split_info["border"])
    if "threshold" in split_info:
        return float(split_info["threshold"])
    raise ValueError("Cannot find threshold/border in CatBoost split info")


def _catboost_split_default_left(split_info, default_left_by_feature=None):
    feature_index = int(split_info.get("float_feature_index", -1))
    if default_left_by_feature is None:
        return True
    return bool(default_left_by_feature.get(feature_index, True))


def _catboost_zero_cover_floor(leaf_weights):
    positive_total = float(np.sum(leaf_weights[leaf_weights > 0.0]))
    floor = positive_total * 1e-12
    return floor if np.isfinite(floor) and floor > 0.0 else 1e-12


def catboost_oblivious_to_simple(
    tree_data,
    scale=1.0,
    flat_feature_index=None,
    default_left_by_feature=None,
):
    """
    Convert one CatBoost oblivious tree from JSON into the simple_tree format.

    CatBoost stores symmetric-tree splits bottom-up and leaf values in
    little-endian leaf-index order. Reversing the splits gives a top-down
    complete binary tree whose BFS leaf order matches CatBoost's leaf order.
    """
    splits = tree_data.get("splits", [])
    leaf_values = np.asarray(tree_data["leaf_values"], dtype=np.float64) * scale
    leaf_weights = np.asarray(
        tree_data.get("leaf_weights", np.ones_like(leaf_values)),
        dtype=np.float64,
    )

    empty_mask = leaf_weights <= 0.0
    if np.any(empty_mask):
        leaf_weights = leaf_weights.copy()
        leaf_weights[empty_mask] = _catboost_zero_cover_floor(leaf_weights)

    depth = len(splits)
    if depth == 0:
        return simple_tree(
            np.array([-1], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            np.array([0.0], dtype=np.float64),
            0,
            np.array([leaf_weights[0]], dtype=np.float64),
            np.array([leaf_values[0]], dtype=np.float64),
            1,
            np.array([False], dtype=np.bool_),
        )

    splits_topdown = list(reversed(splits))
    num_leaves = 1 << depth
    num_internal = num_leaves - 1
    total_nodes = (1 << (depth + 1)) - 1

    # Symmetric trees use one split per depth. Validate and map that split
    # once, then expand complete BFS levels with NumPy instead of repeating
    # Python metadata work for all 2^depth - 1 internal nodes.
    level_features = np.asarray(
        [
            _catboost_split_feature(split_info, flat_feature_index)
            for split_info in splits_topdown
        ],
        dtype=np.int64,
    )
    level_thresholds = np.asarray(
        [_catboost_split_threshold(split_info) for split_info in splits_topdown],
        dtype=np.float64,
    )
    level_default_left = np.asarray(
        [
            _catboost_split_default_left(split_info, default_left_by_feature)
            for split_info in splits_topdown
        ],
        dtype=np.bool_,
    )

    children_left = np.full(total_nodes, -1, dtype=np.int64)
    children_right = np.full(total_nodes, -1, dtype=np.int64)
    feature = np.full(total_nodes, -1, dtype=np.int64)
    threshold = np.zeros(total_nodes, dtype=np.float64)
    value = np.zeros(total_nodes, dtype=np.float64)
    n_node_samples = np.zeros(total_nodes, dtype=np.float64)
    default_left = np.zeros(total_nodes, dtype=np.bool_)

    internal_nodes = np.arange(num_internal, dtype=np.int64)
    level_counts = np.left_shift(1, np.arange(depth, dtype=np.int64))
    children_left[:num_internal] = 2 * internal_nodes + 1
    children_right[:num_internal] = 2 * internal_nodes + 2
    feature[:num_internal] = np.repeat(level_features, level_counts)
    threshold[:num_internal] = np.repeat(level_thresholds, level_counts)
    default_left[:num_internal] = np.repeat(level_default_left, level_counts)

    if leaf_values.shape[0] != num_leaves:
        raise ValueError("CatBoost leaf_values length does not match tree depth")

    leaf_slice = slice(num_internal, total_nodes)
    value[leaf_slice] = leaf_values
    n_node_samples[leaf_slice] = leaf_weights

    for level in range(depth - 1, -1, -1):
        nodes = np.arange((1 << level) - 1, (1 << (level + 1)) - 1)
        left = 2 * nodes + 1
        right = left + 1
        nl = n_node_samples[left]
        nr = n_node_samples[right]
        total = nl + nr
        n_node_samples[nodes] = total
        value[nodes] = (nl * value[left] + nr * value[right]) / total

    return simple_tree(
        children_left,
        children_right,
        feature,
        threshold,
        depth,
        n_node_samples,
        value,
        total_nodes,
        default_left,
    )


def catboost_non_oblivious_to_simple(
    tree_data,
    scale=1.0,
    flat_feature_index=None,
    default_left_by_feature=None,
):
    """Convert one nested Depthwise/Lossguide CatBoost tree to ``simple_tree``."""
    nodes = []
    depths = []

    def append_node(node_data, depth):
        if not isinstance(node_data, dict):
            raise ValueError("CatBoost tree nodes must be JSON objects")

        node = len(nodes)
        nodes.append(node_data)
        depths.append(depth)

        is_leaf = "value" in node_data and "split" not in node_data
        if is_leaf:
            return node
        if not all(key in node_data for key in ("split", "left", "right")):
            raise ValueError(
                "Malformed non-symmetric CatBoost tree: expected split, left, and right"
            )

        left = append_node(node_data["left"], depth + 1)
        right = append_node(node_data["right"], depth + 1)
        node_data = dict(node_data)
        node_data["_qshap_left"] = left
        node_data["_qshap_right"] = right
        nodes[node] = node_data
        return node

    append_node(tree_data, 0)
    node_count = len(nodes)
    raw_leaf_weights = np.asarray(
        [
            float(node_data.get("weight", 1.0))
            for node_data in nodes
            if "_qshap_left" not in node_data
        ],
        dtype=np.float64,
    )
    zero_cover_floor = _catboost_zero_cover_floor(raw_leaf_weights)
    children_left = np.full(node_count, -1, dtype=np.int64)
    children_right = np.full(node_count, -1, dtype=np.int64)
    feature = np.full(node_count, -1, dtype=np.int64)
    threshold = np.zeros(node_count, dtype=np.float64)
    value = np.zeros(node_count, dtype=np.float64)
    n_node_samples = np.zeros(node_count, dtype=np.float64)
    default_left = np.zeros(node_count, dtype=np.bool_)

    for node, node_data in enumerate(nodes):
        if "_qshap_left" not in node_data:
            leaf_value = np.asarray(node_data.get("value"), dtype=np.float64)
            if leaf_value.ndim != 0:
                raise NotImplementedError(
                    "Only single-output CatBoost regression trees are supported"
                )
            value[node] = float(leaf_value) * scale
            # Empty CatBoost leaves need a positive cover for TreeSHAP's
            # conditional expectations.  Keep their prediction unchanged.
            leaf_weight = float(node_data.get("weight", 1.0))
            n_node_samples[node] = (
                leaf_weight if leaf_weight > 0.0 else zero_cover_floor
            )
            continue

        split_info = node_data["split"]
        children_left[node] = int(node_data["_qshap_left"])
        children_right[node] = int(node_data["_qshap_right"])
        feature[node] = _catboost_split_feature(split_info, flat_feature_index)
        threshold[node] = _catboost_split_threshold(split_info)
        default_left[node] = _catboost_split_default_left(
            split_info, default_left_by_feature
        )

    for node in range(node_count - 1, -1, -1):
        left = children_left[node]
        if left < 0:
            continue
        right = children_right[node]
        nl = n_node_samples[left]
        nr = n_node_samples[right]
        total = nl + nr
        n_node_samples[node] = total
        value[node] = (nl * value[left] + nr * value[right]) / total

    return simple_tree(
        children_left,
        children_right,
        feature,
        threshold,
        max(depths, default=0),
        n_node_samples,
        value,
        node_count,
        default_left,
    )


def catboost_formatter(model_data):
    """
    Convert CatBoost JSON model data into simple_tree objects.

    Returns:
    - trees: list[simple_tree]
    - bias: CatBoost model bias/intercept
    - max_depth: maximum depth across trees
    """
    scale = 1.0
    bias = 0.0
    scale_and_bias = model_data.get("scale_and_bias")
    if scale_and_bias is not None and len(scale_and_bias) >= 2:
        scale = float(scale_and_bias[0])
        raw_bias = scale_and_bias[1]
        if isinstance(raw_bias, list):
            if len(raw_bias) > 1:
                raise NotImplementedError(
                    "Only single-output CatBoost regression models are supported"
                )
            bias = float(raw_bias[0]) if raw_bias else 0.0
        else:
            bias = float(raw_bias)

    flat_feature_index, default_left_by_feature = _catboost_float_feature_metadata(
        model_data
    )

    trees_data = model_data.get("oblivious_trees")
    if trees_data is not None:
        trees = [
            catboost_oblivious_to_simple(
                tree,
                scale=scale,
                flat_feature_index=flat_feature_index,
                default_left_by_feature=default_left_by_feature,
            )
            for tree in trees_data
        ]
    elif model_data.get("trees") is not None:
        trees = [
            catboost_non_oblivious_to_simple(
                tree,
                scale=scale,
                flat_feature_index=flat_feature_index,
                default_left_by_feature=default_left_by_feature,
            )
            for tree in model_data["trees"]
        ]
    else:
        raise ValueError(
            "Could not find CatBoost trees in JSON (expected oblivious_trees or trees)"
        )

    max_depth = max((tree.max_depth for tree in trees), default=0)
    return trees, bias, max_depth


def simple_trees_to_shap_models(formatter):
    """
    transform simple_tree objects into the dictionary format shap.TreeExplainer accepts

    Parameters:
    -formatter: output from lgb_formatter(), catboost_formatter(), or similar

    Return:
    A list of one-tree models that shap.TreeExplainer can call
    """
    num_tree = len(formatter)
    shap_models = []
    
    for i in range(num_tree):
        tree = formatter[i]
        children_left = tree.children_left
        children_right = tree.children_right
        children_default = np.where(tree.default_left, children_left, children_right)
        features = tree.feature
        thresholds = tree.threshold
        values = tree.value.reshape(tree.value.shape[0], 1)
        node_sample_weight = tree.n_node_samples
    
        tree_dict = {
        "children_left": children_left,
        "children_right": children_right,
        "children_default": children_default,
        "features": features,
        "thresholds": thresholds,
        "values": values,
        "node_sample_weight": node_sample_weight,
        }
        model = {"trees": [tree_dict]}

        shap_models.append(model)
        
    return(shap_models)


def lgb_shap(formatter):
    return simple_trees_to_shap_models(formatter)

# Define a function to divide the dataset into chunks
def divide_chunks(data, n_chunks):
    total_elements = data.shape[0]
    chunk_size = total_elements // n_chunks
    remainder = total_elements % n_chunks

    chunks = []
    for i in range(n_chunks):
        start_index = i * chunk_size
        # For the last chunk, add the remainder to it
        end_index = start_index + chunk_size + (remainder if i == n_chunks - 1 else 0)
        chunks.append(data[start_index:end_index])

    return chunks
