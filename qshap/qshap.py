import numpy as np
from numba import njit
#import cProfile  
from numba import complex128, int32
from qshap.utils import inv_binom_coef, complex_v_invc_degree, store_complex_v_invc, complex_dot_v2, weight, store_complex_root


@njit
def T2_sample(i, w_matrix, w_ind, init_prediction, store_v_invc, store_z, shap_value, feature_uniq):
    ## Calculate T2 for each sample
    L, p = w_matrix.shape
     
    for l1 in range(L):
        for l2 in range(l1, L):
            init_prediction_product = init_prediction[l1] * init_prediction[l2]
            
            union_f12 = feature_uniq[(w_ind[l1, feature_uniq] + w_ind[l2, feature_uniq]) >= 1]
            
            n12 = len(union_f12)
            # begin to use the property of complex conjugate
            n12_c = n12 // 2 + 1

            v_invc = store_v_invc[n12, :n12_c]
            z = store_z[n12, :n12_c]

            p_z = np.zeros((n12_c), dtype=complex128)
            tmp_p_z = np.zeros((n12_c), dtype=complex128)

            for k in range(n12_c):
                p_z[k] = np.prod(z[k] + w_matrix[l1, union_f12] * w_matrix[l2, union_f12])

            # update only when feature j belongs to the union of f1 and f2
            for j in union_f12:
                    # remove the operation for j by dividing
                for k in range(n12_c):
                    tmp_p_z[k] = p_z[k] / (z[k] + w_matrix[l1, j] * w_matrix[l2, j])

                if l1 != l2:
                    shap_value[i, j] += 2 * (w_matrix[l1, j] * w_matrix[l2, j] - 1) * init_prediction_product * complex_dot_v2(tmp_p_z, v_invc, n12)
                else:
                    shap_value[i, j] += (w_matrix[l1, j] * w_matrix[l2, j] - 1) * init_prediction_product * complex_dot_v2(tmp_p_z, v_invc, n12)

                    
def T2(x, summary_tree, store_v_invc, store_z, parallel = True):
    """
    Calculate the second order treeshap value
    
    Parameters:
    -x: sample to be explained
    -summary_tree: summary tree
    
    Return:
    treeshap value for the sample
    """
    
    init_prediction = summary_tree.init_prediction[summary_tree.init_prediction!=0]
    
    shap_value = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        xi = x[i]
        w = weight(xi, summary_tree)
        w_matrix = w[0]
        w_ind = w[1]

        T2_sample(i, w_matrix, w_ind, init_prediction, store_v_invc, store_z, shap_value, summary_tree.feature_uniq)

    return shap_value


def loss_treeshap(x, y, summary_tree, store_v_invc, store_z, explainer, learning_rate=1):
    """
    Explain l2 loss for every sample
    
    Parameters:
    -x: samples
    -y: y corresponding to x
    -summary_tree: summary tree
    -store_v_invc: stored v_invc/d
    -store_z: stored complex root of unity 
    -explainer: explainer from shap
    -learning_rate: learning_rate if it's a tree from scikit learn GBM, learning_rate for decision tree and xgboost should be 1. 
    
    Return:
    loss treeshap for x
    """

    square_treeshap_x = T2(x, summary_tree, store_v_invc, store_z) * learning_rate ** 2 
    # direct call from shap
    T0_x = explainer.shap_values(x) * learning_rate 
    res = square_treeshap_x - 2 * (y * T0_x.T).T
    return res

# WARNING: This is the wrong way to parallelize the code! The overhead is even more severe for ensemble trees, and it only serves as a bad example to warn the successors to NOT parallelize in this way.
# def loss_treeshap_parallel(x, y, summary_tree, store_v_invc, store_z, explainer, learning_rate=1, ncore=-1):
#     """
#     Explain l2 loss for every sample
    
#         Explain l2 loss for every sample
    
#     Parameters:
#     -x: samples
#     -y: y corresponding to x
#     -summary_tree: summary tree
#     -store_v_invc: stored v_invc/d
#     -store_z: stored complex root of unity 
#     -explainer: explainer from shap
#     -learning_rate: learning_rate if it's a tree from scikit learn GBM, learning_rate for decision tree and xgboost should be 1. 
#     -ncore: number of cores to use, with default value -1 to utilize all the cores
    
#     Return:
#     loss treeshap for x
#     """
#     max_core = os.cpu_count()
#     if ncore == -1:
#         ncore = os.cpu_count()
#     ncore = min(max_core, ncore)
    
#     if ncore==1:
#         square_treeshap_x = T2(x, summary_tree, store_v_invc, store_z) * learning_rate ** 2 
#     else:
#         chunks = divide_chunks(x, ncore)

#         # Use ProcessPoolExecutor to process each chunk in parallel
#         with ProcessPoolExecutor(max_workers=ncore) as executor:
#             # Submit all chunks for processing
#             futures = [executor.submit(T2, chunk, summary_tree, store_v_invc, store_z) for chunk in chunks]

#             # Wait for all futures to complete and collect results
#             results = [future.result() for future in futures]

#         # Combine the results back into a single array
#         # have to compensate the learning_rate
#         square_treeshap_x = np.concatenate(results) * learning_rate ** 2      
        
#     # direct call from shap
#     T0_x = explainer.shap_values(x) * learning_rate 
#     res = square_treeshap_x - 2 * (y * T0_x.T).T
#     return res

