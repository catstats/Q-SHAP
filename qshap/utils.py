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
    node_count: int
    

    
def summarize_tree(tree):
    """
    Summarize the data needed for tree_summary. The tree object should have:
    children_left: left children_index
    children_right: right children_index
    feature: array of features splitted at each node
    threshold: array of threshols for corresponding splitting features
    max_depth: max_depth of the tree
    n_node_samples: array of sample size for each node
    value: array of values for each node, only leaf value is used, so only keep leaf value is fine
    node_count: total number of leaves 
    """
    sample_weight = np.ones_like(tree.threshold)
    init_prediction = np.zeros_like(tree.threshold)
    n = tree.n_node_samples[0]
    
    def traversal_summarize_tree(v):
        v_l, v_r = tree.children_left[v], tree.children_right[v]
        n_v, n_l, n_r = tree.n_node_samples[v], tree.n_node_samples[v_l], tree.n_node_samples[v_r]
        
        if v_l < 0:  #leaf
            init_prediction[v] = tree.value[v] * n_v/n
        else:
            sample_weight[v_l], sample_weight[v_r] = n_v/n_l, n_v/n_r
            traversal_summarize_tree(v_l)
            traversal_summarize_tree(v_r)
    
    # travel from the root
    traversal_summarize_tree(0)
    
    feature_uniq = np.unique(tree.feature[tree.feature >= 0])
   
    return tree_summary(tree.children_left, tree.children_right, tree.feature, feature_uniq, tree.threshold, tree.max_depth, sample_weight, init_prediction, tree.node_count)


def traversal_weight(x, v, w, children_left, children_right, feature, threshold, sample_weight, leaf_ind, w_res, w_ind, depth, met_feature):
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

        if x[split_feature] <= split_threshold:
            w[depth] = w[former_depth] * sample_weight[v_l]
            w_r[depth] = 0
        else:
            w_r[depth] = w_r[former_depth] * sample_weight[v_r]
            w[depth] = 0 

        traversal_weight(x, v_l, w, children_left, children_right, feature, threshold, sample_weight, leaf_ind, w_res, w_ind, depth+1, met_feature) 
        traversal_weight(x, v_r, w_r, children_left, children_right, feature, threshold, sample_weight, leaf_ind, w_res, w_ind, depth+1, met_feature)

        
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
    traversal_weight(x, 0, w, summary_tree.children_left, summary_tree.children_right, summary_tree.feature, summary_tree.threshold, summary_tree.sample_weight, leaf_ind, w_res, w_ind, 0, met_feature)
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
        xgb_tree.append(simple_tree(np.array(tree["left_children"]),
                                    np.array(tree["right_children"]), 
                                    np.array(tree["split_indices"]),
                                    np.array(tree["split_conditions"]),
                                    max_depth, 
                                    np.array(tree["sum_hessian"]), 
                                    np.array(tree["base_weights"]), 
                                    int(tree["tree_param"]["num_nodes"])))

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

    for tree_id in range(ntree):
        
        tree = model_data[model_data['tree_index']==tree_id] 

        node_mapping = {original: idx for idx, original in enumerate(tree['node_index'])}
        node_mapping[None] = - 1

        split_feature = np.array([int(f.replace("Column_", "")) if f is not None else -1 for f in tree['split_feature']])
        
        lgb_tree.append(simple_tree(np.array(tree['left_child'].map(node_mapping)),
                                    np.array(tree['right_child'].map(node_mapping)), 
                                    split_feature,
                                    np.array(tree["threshold"]),
                                    max_depth, 
                                    np.array(tree["count"]), 
                                    np.array(tree["value"]), 
                                    int(tree.shape[0])))
    return(lgb_tree)


def lgb_shap(formatter):
    """
    transform the output from lgb_formatter() so that is necessary for Treeshap calculation
    
    Parameters:
    -formater: output from lgb_formatter()

    Return:
    A new model Treeshap can call
    """
    num_tree = len(formatter)
    lgb_shap = []
    
    for i in range(num_tree):
        tree = formatter[i]
        children_left = tree.children_left
        children_right = tree.children_right
        children_default = children_right.copy()  # because sklearn does not use missing values
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

        lgb_shap.append(model)
        
    return(lgb_shap)

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

