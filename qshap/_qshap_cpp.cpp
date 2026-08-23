#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "catboost_fused_router.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct ArrayRef {
    PyArrayObject *ptr;

    ArrayRef(PyObject *obj, int typenum) : ptr(nullptr) {
        ptr = reinterpret_cast<PyArrayObject *>(
            PyArray_FROM_OTF(obj, typenum, NPY_ARRAY_IN_ARRAY));
        if (ptr == nullptr) {
            throw std::runtime_error("Could not convert input to a contiguous NumPy array");
        }
    }

    ~ArrayRef() {
        Py_XDECREF(ptr);
    }

    ArrayRef(const ArrayRef &) = delete;
    ArrayRef &operator=(const ArrayRef &) = delete;
};

struct PyObjectRef {
    PyObject *ptr;

    explicit PyObjectRef(PyObject *obj) : ptr(obj) {}

    ~PyObjectRef() {
        Py_XDECREF(ptr);
    }

    PyObjectRef(const PyObjectRef &) = delete;
    PyObjectRef &operator=(const PyObjectRef &) = delete;
};

inline void require_ndim(const ArrayRef &arr, int ndim, const char *name) {
    if (PyArray_NDIM(arr.ptr) != ndim) {
        throw std::runtime_error(std::string(name) + " has the wrong number of dimensions");
    }
}

inline npy_intp dim(const ArrayRef &arr, int axis) {
    return PyArray_DIM(arr.ptr, axis);
}

inline double *as_double(ArrayRef &arr) {
    return static_cast<double *>(PyArray_DATA(arr.ptr));
}

inline const double *as_double(const ArrayRef &arr) {
    return static_cast<const double *>(PyArray_DATA(arr.ptr));
}

inline const long long *as_int64(const ArrayRef &arr) {
    return static_cast<const long long *>(PyArray_DATA(arr.ptr));
}

inline const npy_cdouble *as_complex128(const ArrayRef &arr) {
    return static_cast<const npy_cdouble *>(PyArray_DATA(arr.ptr));
}

inline std::complex<double> get_complex(const npy_cdouble *data, npy_intp ncol,
                                        int row, int col) {
    const npy_cdouble z = data[row * ncol + col];
    return std::complex<double>(npy_creal(z), npy_cimag(z));
}

struct TreeArrays {
    const long long *children_left;
    const long long *children_right;
    const long long *feature;
    const long long *feature_uniq;
    const double *threshold;
    const double *sample_weight;
    const double *init_prediction;
    const long long *default_left;
    bool xgboost_split;
    int node_count;
    int n_unique_features;
};

inline bool tree_goes_left(double value, int node, const TreeArrays &tree) {
    if (std::isnan(value)) {
        return tree.default_left[node] != 0;
    }
    if (tree.xgboost_split) {
        return static_cast<float>(value) < static_cast<float>(tree.threshold[node]);
    }
    return value <= tree.threshold[node];
}

struct CatBoostTreeArrays {
    const long long *children_left;
    const long long *children_right;
    const long long *feature;
    const long long *default_left;
    const double *threshold;
    const double *n_node_samples;
    const double *value;
    int max_depth;
    int node_count;
};

std::vector<int> leaf_nodes(const TreeArrays &tree) {
    std::vector<int> leaves;
    leaves.reserve(tree.node_count);
    for (int v = 0; v < tree.node_count; ++v) {
        if (tree.children_left[v] < 0) {
            leaves.push_back(v);
        }
    }
    return leaves;
}

void traversal_weight(
    const double *x_row,
    int node,
    int depth,
    std::vector<double> &w,
    std::vector<int> &met_feature,
    const TreeArrays &tree,
    const std::vector<int> &leaf_position,
    std::vector<double> &w_matrix,
    std::vector<int> &w_ind,
    int n_features)
{
    const int left = static_cast<int>(tree.children_left[node]);
    const int right = static_cast<int>(tree.children_right[node]);

    if (left < 0) {
        const int leaf_row = leaf_position[node];
        for (int d = 0; d < depth; ++d) {
            const int f = met_feature[d];
            if (f >= 0) {
                w_matrix[leaf_row * n_features + f] = w[d];
                w_ind[leaf_row * n_features + f] = 1;
            }
        }
        return;
    }

    const int split_feature = static_cast<int>(tree.feature[node]);
    int former_depth = -1;
    for (int d = 0; d < depth; ++d) {
        if (met_feature[d] == split_feature) {
            former_depth = d;
        }
    }
    const double base_weight = (former_depth < 0) ? 1.0 : w[former_depth];

    const double old_weight = w[depth];
    const int old_feature = met_feature[depth];
    met_feature[depth] = split_feature;

    if (tree_goes_left(x_row[split_feature], node, tree)) {
        w[depth] = base_weight * tree.sample_weight[left];
        traversal_weight(x_row, left, depth + 1, w, met_feature, tree, leaf_position,
                         w_matrix, w_ind, n_features);

        w[depth] = 0.0;
        traversal_weight(x_row, right, depth + 1, w, met_feature, tree,
                         leaf_position, w_matrix, w_ind, n_features);
    } else {
        w[depth] = 0.0;
        traversal_weight(x_row, left, depth + 1, w, met_feature, tree, leaf_position,
                         w_matrix, w_ind, n_features);

        w[depth] = base_weight * tree.sample_weight[right];
        traversal_weight(x_row, right, depth + 1, w, met_feature, tree,
                         leaf_position, w_matrix, w_ind, n_features);
    }

    w[depth] = old_weight;
    met_feature[depth] = old_feature;
}

void compute_weight(
    const double *x_row,
    int n_features,
    const TreeArrays &tree,
    const std::vector<int> &leaves,
    const std::vector<int> &leaf_position,
    std::vector<double> &w_matrix,
    std::vector<int> &w_ind)
{
    const int n_leaves = static_cast<int>(leaves.size());
    w_matrix.assign(static_cast<size_t>(n_leaves) * n_features, 0.0);
    w_ind.assign(static_cast<size_t>(n_leaves) * n_features, 0);

    for (int row = 0; row < n_leaves; ++row) {
        for (int k = 0; k < tree.n_unique_features; ++k) {
            const int f = static_cast<int>(tree.feature_uniq[k]);
            w_matrix[row * n_features + f] = 1.0;
        }
    }

    std::vector<double> w(tree.node_count, 0.0);
    std::vector<int> met_feature(tree.node_count, -1);
    traversal_weight(x_row, 0, 0, w, met_feature, tree, leaf_position,
                     w_matrix, w_ind, n_features);
}

std::string decision_signature(
    const double *x_row,
    const TreeArrays &tree)
{
    std::string sig(static_cast<size_t>(tree.node_count), '\0');
    for (int v = 0; v < tree.node_count; ++v) {
        if (tree.children_left[v] >= 0) {
            const int f = static_cast<int>(tree.feature[v]);
            sig[static_cast<size_t>(v)] = tree_goes_left(x_row[f], v, tree) ? '\1' : '\0';
        }
    }
    return sig;
}

void t2_sample(
    int row,
    const std::vector<double> &w_matrix,
    const std::vector<int> &w_ind,
    const std::vector<double> &leaf_init_prediction,
    const npy_cdouble *store_v_invc,
    const npy_cdouble *store_z,
    npy_intp store_cols,
    std::vector<double> &shap_value,
    const TreeArrays &tree,
    int n_features)
{
    const int n_leaves = static_cast<int>(leaf_init_prediction.size());
    const double eps2 = 1e-18;

    // Generic T2 uses the Q-SHAP complex root-of-unity polynomial basis
    // supplied through store_z/store_v_invc.
    std::vector<int> union_feats;
    std::vector<std::complex<double>> pz;
    union_feats.reserve(static_cast<size_t>(tree.n_unique_features));

    for (int l1 = 0; l1 < n_leaves; ++l1) {
        for (int l2 = l1; l2 < n_leaves; ++l2) {
            const double init_prod = leaf_init_prediction[l1] * leaf_init_prediction[l2];

            union_feats.clear();
            for (int k = 0; k < tree.n_unique_features; ++k) {
                const int f = static_cast<int>(tree.feature_uniq[k]);
                if (w_ind[l1 * n_features + f] + w_ind[l2 * n_features + f] >= 1) {
                    union_feats.push_back(f);
                }
            }

            const int n12 = static_cast<int>(union_feats.size());
            if (n12 == 0) {
                continue;
            }
            const int n12_c = n12 / 2 + 1;

            pz.assign(static_cast<size_t>(n12_c), std::complex<double>(1.0, 0.0));
            for (int k = 0; k < n12_c; ++k) {
                std::complex<double> prod(1.0, 0.0);
                const std::complex<double> zk = get_complex(store_z, store_cols, n12, k);
                for (const int f : union_feats) {
                    const double a = w_matrix[l1 * n_features + f];
                    const double b = w_matrix[l2 * n_features + f];
                    prod *= (zk + a * b);
                }
                pz[k] = prod;
            }

            for (const int j : union_feats) {
                const double a = w_matrix[l1 * n_features + j];
                const double b = w_matrix[l2 * n_features + j];
                const double ab = a * b;
                const double w_factor = ab - 1.0;

                const std::complex<double> denom0 =
                    get_complex(store_z, store_cols, n12, 0) + ab;
                std::complex<double> acc =
                    (pz[0] / denom0) * get_complex(store_v_invc, store_cols, n12, 0);

                if (n12 % 2 == 0) {
                    for (int k = 1; k < n12_c - 1; ++k) {
                        const std::complex<double> denom =
                            get_complex(store_z, store_cols, n12, k) + ab;
                        if (std::norm(denom) >= eps2) {
                            acc += 2.0 * (pz[k] / denom) *
                                   get_complex(store_v_invc, store_cols, n12, k);
                        }
                    }
                    const int k = n12_c - 1;
                    const std::complex<double> denom =
                        get_complex(store_z, store_cols, n12, k) + ab;
                    if (std::norm(denom) >= eps2) {
                        acc += (pz[k] / denom) *
                               get_complex(store_v_invc, store_cols, n12, k);
                    }
                } else {
                    for (int k = 1; k < n12_c; ++k) {
                        const std::complex<double> denom =
                            get_complex(store_z, store_cols, n12, k) + ab;
                        if (std::norm(denom) >= eps2) {
                            acc += 2.0 * (pz[k] / denom) *
                                   get_complex(store_v_invc, store_cols, n12, k);
                        }
                    }
                }

                const double final_contribution = w_factor * acc.real() * init_prod;
                shap_value[row * n_features + j] +=
                    (l1 == l2) ? final_contribution : 2.0 * final_contribution;
            }
        }
    }
}

std::vector<double> compute_t2_values(
    const double *x,
    int n_samples,
    int n_features,
    const TreeArrays &tree,
    const npy_cdouble *store_v_invc,
    const npy_cdouble *store_z,
    npy_intp store_cols)
{
    const std::vector<int> leaves = leaf_nodes(tree);
    if (leaves.empty()) {
        throw std::runtime_error("Tree has no leaves");
    }

    std::vector<double> leaf_init_prediction;
    leaf_init_prediction.reserve(leaves.size());
    for (const int node : leaves) {
        leaf_init_prediction.push_back(tree.init_prediction[node]);
    }

    std::vector<int> leaf_position(static_cast<size_t>(tree.node_count), -1);
    for (int i = 0; i < static_cast<int>(leaves.size()); ++i) {
        leaf_position[leaves[i]] = i;
    }

    std::unordered_map<std::string, std::vector<int>> groups;
    groups.reserve(static_cast<size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        const double *x_row = x + static_cast<size_t>(i) * n_features;
        groups[decision_signature(x_row, tree)].push_back(i);
    }

    std::vector<double> shap_value(static_cast<size_t>(n_samples) * n_features, 0.0);
    std::vector<double> w_matrix;
    std::vector<int> w_ind;

    for (const auto &entry : groups) {
        const int representative = entry.second.front();
        const double *x_row = x + static_cast<size_t>(representative) * n_features;

        compute_weight(x_row, n_features, tree, leaves, leaf_position, w_matrix, w_ind);
        t2_sample(representative, w_matrix, w_ind, leaf_init_prediction,
                  store_v_invc, store_z, store_cols, shap_value, tree, n_features);

        const double *source = shap_value.data() +
                               static_cast<size_t>(representative) * n_features;
        for (size_t k = 1; k < entry.second.size(); ++k) {
            double *target = shap_value.data() +
                             static_cast<size_t>(entry.second[k]) * n_features;
            std::copy(source, source + n_features, target);
        }
    }

    return shap_value;
}

TreeArrays make_tree_arrays(
    const ArrayRef &children_left,
    const ArrayRef &children_right,
    const ArrayRef &feature,
    const ArrayRef &feature_uniq,
    const ArrayRef &threshold,
    const ArrayRef &sample_weight,
    const ArrayRef &init_prediction,
    const ArrayRef &default_left,
    const ArrayRef &xgboost_split)
{
    require_ndim(children_left, 1, "children_left");
    require_ndim(children_right, 1, "children_right");
    require_ndim(feature, 1, "feature");
    require_ndim(feature_uniq, 1, "feature_uniq");
    require_ndim(threshold, 1, "threshold");
    require_ndim(sample_weight, 1, "sample_weight");
    require_ndim(init_prediction, 1, "init_prediction");
    require_ndim(default_left, 1, "default_left");
    require_ndim(xgboost_split, 1, "xgboost_split");

    const int node_count = static_cast<int>(dim(children_left, 0));
    if (dim(children_right, 0) != node_count || dim(feature, 0) != node_count ||
        dim(threshold, 0) != node_count || dim(sample_weight, 0) != node_count ||
        dim(init_prediction, 0) != node_count || dim(default_left, 0) != node_count) {
        throw std::runtime_error("Tree arrays must all have the same node_count");
    }
    if (dim(xgboost_split, 0) != 1) {
        throw std::runtime_error("xgboost_split must contain one value");
    }

    TreeArrays tree;
    tree.children_left = as_int64(children_left);
    tree.children_right = as_int64(children_right);
    tree.feature = as_int64(feature);
    tree.feature_uniq = as_int64(feature_uniq);
    tree.threshold = as_double(threshold);
    tree.sample_weight = as_double(sample_weight);
    tree.init_prediction = as_double(init_prediction);
    tree.default_left = as_int64(default_left);
    tree.xgboost_split = as_int64(xgboost_split)[0] != 0;
    tree.node_count = node_count;
    tree.n_unique_features = static_cast<int>(dim(feature_uniq, 0));
    return tree;
}

PyObject *make_output_array(const std::vector<double> &values, int n_samples, int n_features) {
    npy_intp dims[2] = {n_samples, n_features};
    PyObject *out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (out == nullptr) {
        return nullptr;
    }
    std::copy(values.begin(), values.end(),
              static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(out))));
    return out;
}

PyObject *make_output_vector(const std::vector<double> &values) {
    npy_intp dims[1] = {static_cast<npy_intp>(values.size())};
    PyObject *out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (out == nullptr) {
        return nullptr;
    }
    std::copy(values.begin(), values.end(),
              static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(out))));
    return out;
}

PyObject *get_attr_checked(PyObject *obj, const char *name) {
    PyObject *attr = PyObject_GetAttrString(obj, name);
    if (attr == nullptr) {
        PyErr_Clear();
        throw std::runtime_error(std::string("CatBoost tree is missing attribute: ") + name);
    }
    return attr;
}

int get_int_attr_checked(PyObject *obj, const char *name) {
    PyObjectRef attr(get_attr_checked(obj, name));
    const long value = PyLong_AsLong(attr.ptr);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw std::runtime_error(std::string("CatBoost tree attribute is not an integer: ") + name);
    }
    return static_cast<int>(value);
}

CatBoostTreeArrays make_catboost_tree_arrays(
    const ArrayRef &children_left,
    const ArrayRef &children_right,
    const ArrayRef &feature,
    const ArrayRef &default_left,
    const ArrayRef &threshold,
    const ArrayRef &n_node_samples,
    const ArrayRef &value,
    int max_depth,
    int node_count)
{
    require_ndim(children_left, 1, "children_left");
    require_ndim(children_right, 1, "children_right");
    require_ndim(feature, 1, "feature");
    require_ndim(default_left, 1, "default_left");
    require_ndim(threshold, 1, "threshold");
    require_ndim(n_node_samples, 1, "n_node_samples");
    require_ndim(value, 1, "value");

    if (dim(children_left, 0) != node_count || dim(children_right, 0) != node_count ||
        dim(feature, 0) != node_count || dim(threshold, 0) != node_count ||
        dim(default_left, 0) != node_count ||
        dim(n_node_samples, 0) != node_count || dim(value, 0) != node_count) {
        throw std::runtime_error("CatBoost tree arrays must all have node_count entries");
    }

    CatBoostTreeArrays tree;
    tree.children_left = as_int64(children_left);
    tree.children_right = as_int64(children_right);
    tree.feature = as_int64(feature);
    tree.default_left = as_int64(default_left);
    tree.threshold = as_double(threshold);
    tree.n_node_samples = as_double(n_node_samples);
    tree.value = as_double(value);
    tree.max_depth = max_depth;
    tree.node_count = node_count;
    return tree;
}

int popcount_int(int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

double choose_double(int n, int k) {
    if (k < 0 || k > n) {
        return 0.0;
    }
    if (k == 0 || k == n) {
        return 1.0;
    }
    k = std::min(k, n - k);
    double out = 1.0;
    for (int i = 1; i <= k; ++i) {
        out *= static_cast<double>(n - k + i);
        out /= static_cast<double>(i);
    }
    return out;
}

std::vector<int> catboost_path_bits(int leaf_id, int depth) {
    std::vector<int> bits(static_cast<size_t>(depth), 0);
    for (int level = 0; level < depth; ++level) {
        bits[static_cast<size_t>(level)] = (leaf_id >> (depth - 1 - level)) & 1;
    }
    return bits;
}

std::unordered_map<int, int> catboost_feature_positions(const std::vector<int> &features) {
    std::unordered_map<int, int> positions;
    positions.reserve(features.size());
    for (int i = 0; i < static_cast<int>(features.size()); ++i) {
        positions[features[static_cast<size_t>(i)]] = i;
    }
    return positions;
}

std::vector<int> catboost_project_masks(
    const std::vector<int> &curr_features,
    const std::vector<int> &next_features)
{
    const int curr_n = 1 << static_cast<int>(curr_features.size());
    std::vector<int> projection(static_cast<size_t>(curr_n), 0);
    const auto curr_pos = catboost_feature_positions(curr_features);

    for (int mask = 0; mask < curr_n; ++mask) {
        int projected = 0;
        for (int next_idx = 0; next_idx < static_cast<int>(next_features.size()); ++next_idx) {
            const auto it = curr_pos.find(next_features[static_cast<size_t>(next_idx)]);
            if (it != curr_pos.end() && (mask & (1 << it->second))) {
                projected |= (1 << next_idx);
            }
        }
        projection[static_cast<size_t>(mask)] = projected;
    }
    return projection;
}

void catboost_leaf_T0_T2_cpp(
    const CatBoostTreeArrays &tree,
    int leaf_id,
    int n_features,
    std::vector<int> &feature_ids,
    std::vector<double> &T0_leaf,
    std::vector<double> &T2_leaf)
{
    const int depth = tree.max_depth;
    feature_ids.clear();
    T0_leaf.clear();
    T2_leaf.clear();
    if (depth == 0) {
        return;
    }

    const int num_leaves = 1 << depth;
    const int num_internal = num_leaves - 1;
    if (tree.node_count != (2 * num_leaves - 1)) {
        throw std::runtime_error(
            "Fast CatBoost Q-SHAP currently supports regression with symmetric numeric trees.");
    }
    for (int node = 0; node < num_internal; ++node) {
        const int split_feature = static_cast<int>(tree.feature[node]);
        if (split_feature < 0 || split_feature >= n_features) {
            throw std::runtime_error(
                "CatBoost split feature index is out of bounds for X columns");
        }
    }

    std::vector<int> level_feature(static_cast<size_t>(depth));
    for (int level = 0; level < depth; ++level) {
        const int node = (1 << level) - 1;
        level_feature[static_cast<size_t>(level)] = static_cast<int>(tree.feature[node]);
    }

    std::vector<std::vector<int>> suffix_features(static_cast<size_t>(depth + 1));
    suffix_features[static_cast<size_t>(depth)] = std::vector<int>();
    for (int level = depth - 1; level >= 0; --level) {
        suffix_features[static_cast<size_t>(level)] =
            suffix_features[static_cast<size_t>(level + 1)];
        const int feature = level_feature[static_cast<size_t>(level)];
        auto &current = suffix_features[static_cast<size_t>(level)];
        if (std::find(current.begin(), current.end(), feature) == current.end()) {
            current.push_back(feature);
        }
    }

    const std::vector<int> hot_bits = catboost_path_bits(leaf_id, depth);

    std::vector<std::vector<double>> dp_next(
        static_cast<size_t>(num_leaves), std::vector<double>(1, 0.0));
    for (int leaf = 0; leaf < num_leaves; ++leaf) {
        dp_next[static_cast<size_t>(leaf)][0] = tree.value[num_internal + leaf];
    }

    for (int level = depth - 1; level >= 0; --level) {
        const auto &curr_features = suffix_features[static_cast<size_t>(level)];
        const auto &next_features = suffix_features[static_cast<size_t>(level + 1)];
        const int curr_n = 1 << static_cast<int>(curr_features.size());
        const int nodes_this_level = 1 << level;
        const auto curr_pos = catboost_feature_positions(curr_features);
        const int split_feature = level_feature[static_cast<size_t>(level)];
        const int split_pos = curr_pos.at(split_feature);
        const auto projection = catboost_project_masks(curr_features, next_features);

        std::vector<std::vector<double>> dp_curr(
            static_cast<size_t>(nodes_this_level),
            std::vector<double>(static_cast<size_t>(curr_n), 0.0));

        for (int node_pos = 0; node_pos < nodes_this_level; ++node_pos) {
            const int bfs_node = (1 << level) - 1 + node_pos;
            const int left_bfs = static_cast<int>(tree.children_left[bfs_node]);
            const int right_bfs = static_cast<int>(tree.children_right[bfs_node]);
            const int left_pos = 2 * node_pos;
            const int right_pos = 2 * node_pos + 1;

            const double parent_n = std::max(
                tree.n_node_samples[bfs_node], std::numeric_limits<double>::min());
            const double p_left = tree.n_node_samples[left_bfs] / parent_n;
            const double p_right = tree.n_node_samples[right_bfs] / parent_n;

            for (int mask = 0; mask < curr_n; ++mask) {
                const int next_mask = projection[static_cast<size_t>(mask)];
                const bool observed = (mask & (1 << split_pos)) != 0;
                if (observed) {
                    dp_curr[static_cast<size_t>(node_pos)][static_cast<size_t>(mask)] =
                        hot_bits[static_cast<size_t>(level)]
                            ? dp_next[static_cast<size_t>(right_pos)][static_cast<size_t>(next_mask)]
                            : dp_next[static_cast<size_t>(left_pos)][static_cast<size_t>(next_mask)];
                } else {
                    dp_curr[static_cast<size_t>(node_pos)][static_cast<size_t>(mask)] =
                        p_left * dp_next[static_cast<size_t>(left_pos)][static_cast<size_t>(next_mask)] +
                        p_right * dp_next[static_cast<size_t>(right_pos)][static_cast<size_t>(next_mask)];
                }
            }
        }

        dp_next.swap(dp_curr);
    }

    const auto &root_features = suffix_features[0];
    const int k = static_cast<int>(root_features.size());
    if (k == 0) {
        return;
    }

    const auto &values = dp_next[0];
    const int total_masks = 1 << k;
    std::vector<double> tmp_T0(static_cast<size_t>(k), 0.0);
    std::vector<double> tmp_T2(static_cast<size_t>(k), 0.0);
    for (int mask = 0; mask < total_masks; ++mask) {
        if (mask == total_masks - 1) {
            continue;
        }
        const int subset_size = popcount_int(mask);
        const double weight = 1.0 / (static_cast<double>(k) * choose_double(k - 1, subset_size));
        for (int bit = 0; bit < k; ++bit) {
            if (mask & (1 << bit)) {
                continue;
            }
            const int with_feature = mask | (1 << bit);
            tmp_T0[static_cast<size_t>(bit)] +=
                weight * (values[static_cast<size_t>(with_feature)] -
                          values[static_cast<size_t>(mask)]);
            tmp_T2[static_cast<size_t>(bit)] +=
                weight * (
                    values[static_cast<size_t>(with_feature)] *
                    values[static_cast<size_t>(with_feature)] -
                    values[static_cast<size_t>(mask)] *
                    values[static_cast<size_t>(mask)]);
        }
    }

    feature_ids.reserve(static_cast<size_t>(k));
    T0_leaf.reserve(static_cast<size_t>(k));
    T2_leaf.reserve(static_cast<size_t>(k));
    for (int bit = 0; bit < k; ++bit) {
        const int feature_id = root_features[static_cast<size_t>(bit)];
        if (feature_id >= 0 && feature_id < n_features &&
            (tmp_T0[static_cast<size_t>(bit)] != 0.0 ||
             tmp_T2[static_cast<size_t>(bit)] != 0.0)) {
            feature_ids.push_back(feature_id);
            T0_leaf.push_back(tmp_T0[static_cast<size_t>(bit)]);
            T2_leaf.push_back(tmp_T2[static_cast<size_t>(bit)]);
        }
    }
}

void catboost_update_global_stats_cpp(
    const qshap_catboost_core::QuantizedRoutingPlan &routing,
    const qshap_catboost_core::QuantizedTreeRoute &tree_route,
    const double *y,
    int n_samples,
    int n_features,
    const CatBoostTreeArrays &tree,
    std::vector<double> &cumulative_prediction,
    std::vector<double> &loss_sum,
    std::vector<double> &total_loss_by_sample,
    bool compute_sd)
{
    const int depth = tree.max_depth;
    const int num_leaves = 1 << depth;
    const int num_internal = num_leaves - 1;
    if (tree.node_count != (2 * num_leaves - 1)) {
        throw std::runtime_error(
            "Fast CatBoost Q-SHAP currently supports regression with symmetric numeric trees.");
    }
    for (int node = 0; node < num_internal; ++node) {
        const int split_feature = static_cast<int>(tree.feature[node]);
        if (split_feature < 0 || split_feature >= n_features) {
            throw std::runtime_error(
                "CatBoost split feature index is out of bounds for X columns");
        }
    }

    std::vector<int> group_n(static_cast<size_t>(num_leaves), 0);
    std::vector<double> group_sum_r(static_cast<size_t>(num_leaves), 0.0);
    std::vector<std::vector<int>> group_rows;
    std::vector<double> sample_residual;
    if (compute_sd) {
        group_rows.resize(static_cast<size_t>(num_leaves));
        sample_residual.resize(static_cast<size_t>(n_samples));
    }

    auto consume_leaf = [&](int i, int leaf) {
        const double residual = y[i] - cumulative_prediction[static_cast<size_t>(i)];
        group_n[static_cast<size_t>(leaf)] += 1;
        group_sum_r[static_cast<size_t>(leaf)] += residual;
        if (compute_sd) {
            group_rows[static_cast<size_t>(leaf)].push_back(i);
            sample_residual[static_cast<size_t>(i)] = residual;
        }
        cumulative_prediction[static_cast<size_t>(i)] += tree.value[num_internal + leaf];
    };
    qshap_catboost_core::route_tree_tiled(
        routing, tree_route, n_samples, consume_leaf
    );

    std::vector<int> feature_ids;
    std::vector<double> T0_leaf;
    std::vector<double> T2_leaf;
    feature_ids.reserve(static_cast<size_t>(depth));
    T0_leaf.reserve(static_cast<size_t>(depth));
    T2_leaf.reserve(static_cast<size_t>(depth));
    for (int leaf = 0; leaf < num_leaves; ++leaf) {
        const int count = group_n[static_cast<size_t>(leaf)];
        if (count == 0) {
            continue;
        }
        catboost_leaf_T0_T2_cpp(tree, leaf, n_features, feature_ids, T0_leaf, T2_leaf);
        const double m = static_cast<double>(count);
        const double sum_r = group_sum_r[static_cast<size_t>(leaf)];

        for (int idx = 0; idx < static_cast<int>(feature_ids.size()); ++idx) {
            const int j = feature_ids[static_cast<size_t>(idx)];
            const double a = T2_leaf[static_cast<size_t>(idx)];
            const double b = 2.0 * T0_leaf[static_cast<size_t>(idx)];
            if (a == 0.0 && b == 0.0) {
                continue;
            }
            loss_sum[static_cast<size_t>(j)] += m * a - b * sum_r;
            if (compute_sd) {
                for (const int row : group_rows[static_cast<size_t>(leaf)]) {
                    const size_t out_idx =
                        static_cast<size_t>(row) * n_features + j;
                    total_loss_by_sample[out_idx] +=
                        a - b * sample_residual[static_cast<size_t>(row)];
                }
            }
        }
    }
}

PyObject *py_t2(PyObject *, PyObject *args) {
    PyObject *x_obj = nullptr;
    PyObject *children_left_obj = nullptr;
    PyObject *children_right_obj = nullptr;
    PyObject *feature_obj = nullptr;
    PyObject *feature_uniq_obj = nullptr;
    PyObject *threshold_obj = nullptr;
    PyObject *sample_weight_obj = nullptr;
    PyObject *init_prediction_obj = nullptr;
    PyObject *default_left_obj = nullptr;
    PyObject *xgboost_split_obj = nullptr;
    PyObject *store_v_invc_obj = nullptr;
    PyObject *store_z_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOO", &x_obj, &children_left_obj,
                          &children_right_obj, &feature_obj, &feature_uniq_obj,
                          &threshold_obj, &sample_weight_obj, &init_prediction_obj,
                          &default_left_obj, &xgboost_split_obj,
                          &store_v_invc_obj, &store_z_obj)) {
        return nullptr;
    }

    try {
        ArrayRef x_arr(x_obj, NPY_DOUBLE);
        ArrayRef children_left_arr(children_left_obj, NPY_INT64);
        ArrayRef children_right_arr(children_right_obj, NPY_INT64);
        ArrayRef feature_arr(feature_obj, NPY_INT64);
        ArrayRef feature_uniq_arr(feature_uniq_obj, NPY_INT64);
        ArrayRef threshold_arr(threshold_obj, NPY_DOUBLE);
        ArrayRef sample_weight_arr(sample_weight_obj, NPY_DOUBLE);
        ArrayRef init_prediction_arr(init_prediction_obj, NPY_DOUBLE);
        ArrayRef default_left_arr(default_left_obj, NPY_INT64);
        ArrayRef xgboost_split_arr(xgboost_split_obj, NPY_INT64);
        ArrayRef store_v_invc_arr(store_v_invc_obj, NPY_COMPLEX128);
        ArrayRef store_z_arr(store_z_obj, NPY_COMPLEX128);

        require_ndim(x_arr, 2, "x");
        require_ndim(store_v_invc_arr, 2, "store_v_invc");
        require_ndim(store_z_arr, 2, "store_z");

        const int n_samples = static_cast<int>(dim(x_arr, 0));
        const int n_features = static_cast<int>(dim(x_arr, 1));
        const TreeArrays tree = make_tree_arrays(children_left_arr, children_right_arr,
                                                 feature_arr, feature_uniq_arr,
                                                 threshold_arr, sample_weight_arr,
                                                 init_prediction_arr, default_left_arr,
                                                 xgboost_split_arr);

        std::vector<double> values = compute_t2_values(
            as_double(x_arr), n_samples, n_features, tree,
            as_complex128(store_v_invc_arr), as_complex128(store_z_arr),
            dim(store_v_invc_arr, 1));
        return make_output_array(values, n_samples, n_features);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyObject *py_loss_treeshap(PyObject *, PyObject *args) {
    PyObject *x_obj = nullptr;
    PyObject *y_obj = nullptr;
    PyObject *children_left_obj = nullptr;
    PyObject *children_right_obj = nullptr;
    PyObject *feature_obj = nullptr;
    PyObject *feature_uniq_obj = nullptr;
    PyObject *threshold_obj = nullptr;
    PyObject *sample_weight_obj = nullptr;
    PyObject *init_prediction_obj = nullptr;
    PyObject *default_left_obj = nullptr;
    PyObject *xgboost_split_obj = nullptr;
    PyObject *store_v_invc_obj = nullptr;
    PyObject *store_z_obj = nullptr;
    PyObject *t0_obj = nullptr;
    double learning_rate = 1.0;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOO|d", &x_obj, &y_obj,
                          &children_left_obj, &children_right_obj, &feature_obj,
                          &feature_uniq_obj, &threshold_obj, &sample_weight_obj,
                          &init_prediction_obj, &default_left_obj,
                          &xgboost_split_obj, &store_v_invc_obj, &store_z_obj,
                          &t0_obj, &learning_rate)) {
        return nullptr;
    }

    try {
        ArrayRef x_arr(x_obj, NPY_DOUBLE);
        ArrayRef y_arr(y_obj, NPY_DOUBLE);
        ArrayRef children_left_arr(children_left_obj, NPY_INT64);
        ArrayRef children_right_arr(children_right_obj, NPY_INT64);
        ArrayRef feature_arr(feature_obj, NPY_INT64);
        ArrayRef feature_uniq_arr(feature_uniq_obj, NPY_INT64);
        ArrayRef threshold_arr(threshold_obj, NPY_DOUBLE);
        ArrayRef sample_weight_arr(sample_weight_obj, NPY_DOUBLE);
        ArrayRef init_prediction_arr(init_prediction_obj, NPY_DOUBLE);
        ArrayRef default_left_arr(default_left_obj, NPY_INT64);
        ArrayRef xgboost_split_arr(xgboost_split_obj, NPY_INT64);
        ArrayRef store_v_invc_arr(store_v_invc_obj, NPY_COMPLEX128);
        ArrayRef store_z_arr(store_z_obj, NPY_COMPLEX128);
        ArrayRef t0_arr(t0_obj, NPY_DOUBLE);

        require_ndim(x_arr, 2, "x");
        require_ndim(y_arr, 1, "y");
        require_ndim(t0_arr, 2, "T0_x");
        require_ndim(store_v_invc_arr, 2, "store_v_invc");
        require_ndim(store_z_arr, 2, "store_z");

        const int n_samples = static_cast<int>(dim(x_arr, 0));
        const int n_features = static_cast<int>(dim(x_arr, 1));
        if (dim(y_arr, 0) != n_samples || dim(t0_arr, 0) != n_samples ||
            dim(t0_arr, 1) != n_features) {
            throw std::runtime_error("x, y, and T0_x dimensions are inconsistent");
        }

        const TreeArrays tree = make_tree_arrays(children_left_arr, children_right_arr,
                                                 feature_arr, feature_uniq_arr,
                                                 threshold_arr, sample_weight_arr,
                                                 init_prediction_arr, default_left_arr,
                                                 xgboost_split_arr);

        std::vector<double> values = compute_t2_values(
            as_double(x_arr), n_samples, n_features, tree,
            as_complex128(store_v_invc_arr), as_complex128(store_z_arr),
            dim(store_v_invc_arr, 1));

        const double *y = as_double(y_arr);
        const double *t0 = as_double(t0_arr);
        const double lr2 = learning_rate * learning_rate;
        const double c = 2.0 * learning_rate;
        const bool lr_is_one = std::abs(learning_rate - 1.0) <= 1e-12;

        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features; ++j) {
                const size_t idx = static_cast<size_t>(i) * n_features + j;
                if (lr_is_one) {
                    values[idx] -= 2.0 * t0[idx] * y[i];
                } else {
                    values[idx] = lr2 * values[idx] - c * t0[idx] * y[i];
                }
            }
        }

        return make_output_array(values, n_samples, n_features);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyObject *py_catboost_qshap_r2_fast(PyObject *, PyObject *args) {
    PyObject *x_obj = nullptr;
    PyObject *y_obj = nullptr;
    PyObject *trees_obj = nullptr;
    double bias = 0.0;
    int compute_sd_int = 1;
    int return_prediction_int = 0;

    if (!PyArg_ParseTuple(args, "OOOd|pp", &x_obj, &y_obj, &trees_obj, &bias,
                          &compute_sd_int, &return_prediction_int)) {
        return nullptr;
    }

    try {
        ArrayRef x_arr(x_obj, NPY_DOUBLE);
        ArrayRef y_arr(y_obj, NPY_DOUBLE);
        require_ndim(x_arr, 2, "X");
        require_ndim(y_arr, 1, "y");

        const int n_samples = static_cast<int>(dim(x_arr, 0));
        const int n_features = static_cast<int>(dim(x_arr, 1));
        if (dim(y_arr, 0) != n_samples) {
            throw std::runtime_error("X and y dimensions are inconsistent");
        }

        PyObjectRef trees_seq(PySequence_Fast(trees_obj, "catboost_trees must be a sequence"));
        if (trees_seq.ptr == nullptr) {
            throw std::runtime_error("catboost_trees must be a sequence");
        }
        const Py_ssize_t num_trees = PySequence_Fast_GET_SIZE(trees_seq.ptr);
        PyObject **tree_items = PySequence_Fast_ITEMS(trees_seq.ptr);

        const double *x = as_double(x_arr);
        const double *y = as_double(y_arr);
        const bool compute_sd = compute_sd_int != 0;

        std::vector<qshap_catboost_core::TreeRouteSpec> route_specs;
        route_specs.reserve(static_cast<size_t>(num_trees));
        std::vector<int> nan_goes_right(static_cast<size_t>(n_features), -1);
        for (Py_ssize_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
            PyObject *tree_obj = tree_items[tree_idx];
            PyObjectRef feature_obj(get_attr_checked(tree_obj, "feature"));
            PyObjectRef threshold_obj(get_attr_checked(tree_obj, "threshold"));
            PyObjectRef default_left_obj(get_attr_checked(tree_obj, "default_left"));
            ArrayRef feature_arr(feature_obj.ptr, NPY_INT64);
            ArrayRef threshold_arr(threshold_obj.ptr, NPY_DOUBLE);
            ArrayRef default_left_arr(default_left_obj.ptr, NPY_INT64);

            const int max_depth = get_int_attr_checked(tree_obj, "max_depth");
            const int node_count = get_int_attr_checked(tree_obj, "node_count");
            if (max_depth < 0 || max_depth >= 30) {
                throw std::runtime_error("Invalid CatBoost symmetric-tree depth");
            }
            const int expected_nodes = 2 * (1 << max_depth) - 1;
            if (node_count != expected_nodes ||
                dim(feature_arr, 0) != node_count ||
                dim(threshold_arr, 0) != node_count ||
                dim(default_left_arr, 0) != node_count) {
                throw std::runtime_error(
                    "Fast CatBoost Q-SHAP requires complete symmetric numeric trees");
            }

            const long long *feature = as_int64(feature_arr);
            const double *threshold = as_double(threshold_arr);
            const long long *default_left = as_int64(default_left_arr);
            qshap_catboost_core::TreeRouteSpec route;
            route.level_feature.resize(static_cast<size_t>(max_depth));
            route.level_border.resize(static_cast<size_t>(max_depth));
            for (int level = 0; level < max_depth; ++level) {
                const int node = (1 << level) - 1;
                const int split_feature = static_cast<int>(feature[node]);
                const float split_border = static_cast<float>(threshold[node]);
            if (split_feature < 0 || split_feature >= n_features) {
                throw std::runtime_error(
                    "CatBoost split feature index is out of bounds for X columns");
            }
            if (!std::isfinite(split_border)) {
                throw std::runtime_error("CatBoost split border is invalid");
            }
                route.level_feature[static_cast<size_t>(level)] = split_feature;
                route.level_border[static_cast<size_t>(level)] = split_border;
                const int missing_right = default_left[node] != 0 ? 0 : 1;
                int &known_direction =
                    nan_goes_right[static_cast<size_t>(split_feature)];
                if (known_direction >= 0 && known_direction != missing_right) {
                    throw std::runtime_error(
                        "CatBoost uses inconsistent NaN directions for one feature");
                }
                known_direction = missing_right;
            }
            route_specs.push_back(std::move(route));
        }
        for (int &direction : nan_goes_right) {
            if (direction < 0) direction = 0;
        }
        const qshap_catboost_core::QuantizedRoutingPlan routing =
            qshap_catboost_core::build_quantized_routing_plan(
                x, n_samples, n_features, false, route_specs, nan_goes_right
            );

        std::vector<double> cumulative_prediction(static_cast<size_t>(n_samples), bias);
        std::vector<double> loss_sum(static_cast<size_t>(n_features), 0.0);
        std::vector<double> loss_sumsq(static_cast<size_t>(n_features), 0.0);
        std::vector<double> total_loss_by_sample;
        if (compute_sd) {
            total_loss_by_sample.assign(
                static_cast<size_t>(n_samples) * n_features, 0.0);
        }

        for (Py_ssize_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
            PyObject *tree_obj = tree_items[tree_idx];

            PyObjectRef children_left_obj(get_attr_checked(tree_obj, "children_left"));
            PyObjectRef children_right_obj(get_attr_checked(tree_obj, "children_right"));
            PyObjectRef feature_obj(get_attr_checked(tree_obj, "feature"));
            PyObjectRef default_left_obj(get_attr_checked(tree_obj, "default_left"));
            PyObjectRef threshold_obj(get_attr_checked(tree_obj, "threshold"));
            PyObjectRef n_node_samples_obj(get_attr_checked(tree_obj, "n_node_samples"));
            PyObjectRef value_obj(get_attr_checked(tree_obj, "value"));

            ArrayRef children_left_arr(children_left_obj.ptr, NPY_INT64);
            ArrayRef children_right_arr(children_right_obj.ptr, NPY_INT64);
            ArrayRef feature_arr(feature_obj.ptr, NPY_INT64);
            ArrayRef default_left_arr(default_left_obj.ptr, NPY_INT64);
            ArrayRef threshold_arr(threshold_obj.ptr, NPY_DOUBLE);
            ArrayRef n_node_samples_arr(n_node_samples_obj.ptr, NPY_DOUBLE);
            ArrayRef value_arr(value_obj.ptr, NPY_DOUBLE);

            const int max_depth = get_int_attr_checked(tree_obj, "max_depth");
            const int node_count = get_int_attr_checked(tree_obj, "node_count");
            const CatBoostTreeArrays tree = make_catboost_tree_arrays(
                children_left_arr, children_right_arr, feature_arr, default_left_arr,
                threshold_arr,
                n_node_samples_arr, value_arr, max_depth, node_count);

            catboost_update_global_stats_cpp(
                routing,
                routing.tree_routes[static_cast<size_t>(tree_idx)],
                y, n_samples, n_features, tree, cumulative_prediction,
                loss_sum, total_loss_by_sample, compute_sd);
        }

        if (compute_sd) {
            for (int i = 0; i < n_samples; ++i) {
                for (int j = 0; j < n_features; ++j) {
                    const double value = total_loss_by_sample[
                        static_cast<size_t>(i) * n_features + j];
                    loss_sumsq[static_cast<size_t>(j)] += value * value;
                }
            }
        }

        double y_sum = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            y_sum += y[i];
        }
        const double y_mean = y_sum / static_cast<double>(n_samples);

        double sst = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            const double centered = y[i] - y_mean;
            sst += centered * centered;
        }
        if (sst <= 0.0) {
            throw std::runtime_error("Cannot compute R2 decomposition when y has zero variance");
        }

        std::vector<double> rsq(static_cast<size_t>(n_features), 0.0);
        for (int j = 0; j < n_features; ++j) {
            rsq[static_cast<size_t>(j)] = -loss_sum[static_cast<size_t>(j)] / sst;
        }

        PyObjectRef out(PyDict_New());
        if (out.ptr == nullptr) {
            return nullptr;
        }

        PyObject *rsq_obj = make_output_vector(rsq);
        PyObject *loss_sum_obj = make_output_vector(loss_sum);
        PyObject *n_obj = PyLong_FromLong(n_samples);
        PyObject *sst_obj = PyFloat_FromDouble(sst);
        if (rsq_obj == nullptr || loss_sum_obj == nullptr || n_obj == nullptr ||
            sst_obj == nullptr) {
            Py_XDECREF(rsq_obj);
            Py_XDECREF(loss_sum_obj);
            Py_XDECREF(n_obj);
            Py_XDECREF(sst_obj);
            return nullptr;
        }

        PyDict_SetItemString(out.ptr, "rsq", rsq_obj);
        PyDict_SetItemString(out.ptr, "loss_sum", loss_sum_obj);
        PyDict_SetItemString(out.ptr, "n", n_obj);
        PyDict_SetItemString(out.ptr, "sst", sst_obj);
        Py_DECREF(rsq_obj);
        Py_DECREF(loss_sum_obj);
        Py_DECREF(n_obj);
        Py_DECREF(sst_obj);

        if (return_prediction_int != 0) {
            PyObject *prediction_obj = make_output_vector(cumulative_prediction);
            if (prediction_obj == nullptr) {
                return nullptr;
            }
            PyDict_SetItemString(out.ptr, "prediction", prediction_obj);
            Py_DECREF(prediction_obj);
        }

        if (compute_sd) {
            std::vector<double> sd_rsq(static_cast<size_t>(n_features),
                                       std::numeric_limits<double>::quiet_NaN());
            if (n_samples > 1) {
                for (int j = 0; j < n_features; ++j) {
                    const double sum = loss_sum[static_cast<size_t>(j)];
                    const double sumsq = loss_sumsq[static_cast<size_t>(j)];
                    const double var = std::max(
                        (sumsq - (sum * sum) / static_cast<double>(n_samples)) /
                            static_cast<double>(n_samples - 1),
                        0.0);
                    sd_rsq[static_cast<size_t>(j)] =
                        std::sqrt(static_cast<double>(n_samples) * var) / sst;
                }
            }

            PyObject *loss_sumsq_obj = make_output_vector(loss_sumsq);
            PyObject *sd_rsq_obj = make_output_vector(sd_rsq);
            if (loss_sumsq_obj == nullptr || sd_rsq_obj == nullptr) {
                Py_XDECREF(loss_sumsq_obj);
                Py_XDECREF(sd_rsq_obj);
                return nullptr;
            }
            PyDict_SetItemString(out.ptr, "loss_sumsq", loss_sumsq_obj);
            PyDict_SetItemString(out.ptr, "sd_rsq", sd_rsq_obj);
            Py_DECREF(loss_sumsq_obj);
            Py_DECREF(sd_rsq_obj);
        } else {
            Py_INCREF(Py_None);
            PyDict_SetItemString(out.ptr, "loss_sumsq", Py_None);
            Py_DECREF(Py_None);
            Py_INCREF(Py_None);
            PyDict_SetItemString(out.ptr, "sd_rsq", Py_None);
            Py_DECREF(Py_None);
        }

        Py_INCREF(out.ptr);
        return out.ptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"t2", py_t2, METH_VARARGS, "Compute second-order tree SHAP values with the C++ backend."},
    {"loss_treeshap", py_loss_treeshap, METH_VARARGS, "Compute Q-SHAP loss values with the C++ backend."},
    {"catboost_qshap_r2_fast", py_catboost_qshap_r2_fast, METH_VARARGS,
     "Compute fast global CatBoost Q-SHAP R2 with the C++ backend."},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_qshap_cpp",
    "C++ backend for qshap.",
    -1,
    methods
};

}  // namespace

PyMODINIT_FUNC PyInit__qshap_cpp(void) {
    import_array();
    return PyModule_Create(&module);
}
