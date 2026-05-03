#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cmath>
#include <complex>
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
    return std::complex<double>(z.real, z.imag);
}

struct TreeArrays {
    const long long *children_left;
    const long long *children_right;
    const long long *feature;
    const long long *feature_uniq;
    const double *threshold;
    const double *sample_weight;
    const double *init_prediction;
    int node_count;
    int n_unique_features;
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
    const double split_threshold = tree.threshold[node];

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

    if (x_row[split_feature] <= split_threshold) {
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
            sig[static_cast<size_t>(v)] = (x_row[f] <= tree.threshold[v]) ? '\1' : '\0';
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
    const ArrayRef &init_prediction)
{
    require_ndim(children_left, 1, "children_left");
    require_ndim(children_right, 1, "children_right");
    require_ndim(feature, 1, "feature");
    require_ndim(feature_uniq, 1, "feature_uniq");
    require_ndim(threshold, 1, "threshold");
    require_ndim(sample_weight, 1, "sample_weight");
    require_ndim(init_prediction, 1, "init_prediction");

    const int node_count = static_cast<int>(dim(children_left, 0));
    if (dim(children_right, 0) != node_count || dim(feature, 0) != node_count ||
        dim(threshold, 0) != node_count || dim(sample_weight, 0) != node_count ||
        dim(init_prediction, 0) != node_count) {
        throw std::runtime_error("Tree arrays must all have the same node_count");
    }

    TreeArrays tree;
    tree.children_left = as_int64(children_left);
    tree.children_right = as_int64(children_right);
    tree.feature = as_int64(feature);
    tree.feature_uniq = as_int64(feature_uniq);
    tree.threshold = as_double(threshold);
    tree.sample_weight = as_double(sample_weight);
    tree.init_prediction = as_double(init_prediction);
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

PyObject *py_t2(PyObject *, PyObject *args) {
    PyObject *x_obj = nullptr;
    PyObject *children_left_obj = nullptr;
    PyObject *children_right_obj = nullptr;
    PyObject *feature_obj = nullptr;
    PyObject *feature_uniq_obj = nullptr;
    PyObject *threshold_obj = nullptr;
    PyObject *sample_weight_obj = nullptr;
    PyObject *init_prediction_obj = nullptr;
    PyObject *store_v_invc_obj = nullptr;
    PyObject *store_z_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOO", &x_obj, &children_left_obj,
                          &children_right_obj, &feature_obj, &feature_uniq_obj,
                          &threshold_obj, &sample_weight_obj, &init_prediction_obj,
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
                                                 init_prediction_arr);

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
    PyObject *store_v_invc_obj = nullptr;
    PyObject *store_z_obj = nullptr;
    PyObject *t0_obj = nullptr;
    double learning_rate = 1.0;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOO|d", &x_obj, &y_obj,
                          &children_left_obj, &children_right_obj, &feature_obj,
                          &feature_uniq_obj, &threshold_obj, &sample_weight_obj,
                          &init_prediction_obj, &store_v_invc_obj, &store_z_obj,
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
                                                 init_prediction_arr);

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

PyMethodDef methods[] = {
    {"t2", py_t2, METH_VARARGS, "Compute second-order tree SHAP values with the C++ backend."},
    {"loss_treeshap", py_loss_treeshap, METH_VARARGS, "Compute Q-SHAP loss values with the C++ backend."},
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
