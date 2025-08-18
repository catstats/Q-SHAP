#include "utils.h"

TreeSummary list_to_tree_summary(const Rcpp::List& tree_summary_list) {
    TreeSummary summary;
    
    Rcpp::IntegerVector children_left_r = tree_summary_list["children_left"];
    Rcpp::IntegerVector children_right_r = tree_summary_list["children_right"];
    Rcpp::IntegerVector feature_r = tree_summary_list["feature"];
    Rcpp::IntegerVector feature_uniq_r = tree_summary_list["feature_uniq"];
    Rcpp::NumericVector threshold_r = tree_summary_list["threshold"];
    Rcpp::NumericVector sample_weight_r = tree_summary_list["sample_weight"];
    Rcpp::NumericVector init_prediction_r = tree_summary_list["init_prediction"];
    
    summary.children_left = Rcpp::as<Eigen::VectorXi>(children_left_r);
    summary.children_right = Rcpp::as<Eigen::VectorXi>(children_right_r);
    summary.feature = Rcpp::as<Eigen::VectorXi>(feature_r);
    summary.feature_uniq = Rcpp::as<Eigen::VectorXi>(feature_uniq_r);
    summary.threshold = Rcpp::as<Eigen::VectorXd>(threshold_r);
    summary.sample_weight = Rcpp::as<Eigen::VectorXd>(sample_weight_r);
    summary.init_prediction = Rcpp::as<Eigen::VectorXd>(init_prediction_r);
    
    summary.max_depth = Rcpp::as<int>(tree_summary_list["max_depth"]);
    summary.node_count = Rcpp::as<int>(tree_summary_list["node_count"]);
    
    return summary;
}

Eigen::VectorXd inv_binom_coef(int d) {
    Eigen::VectorXd coef(d + 1);
    coef(0) = 1;

    for (int i = 1; i < d / 2 + 1; i++) {
        coef(i) = (coef(i - 1) * (d - i + 1)) / i;
    }

    for (int i = d / 2 + 1; i < d + 1; i++) {
        coef(i) = coef(d - i);
    }
    
    return coef.cwiseInverse();
}

Eigen::MatrixXcd complex_v_invc_degree(int d) {
    Eigen::VectorXcd omega_inv(d);
    for (int k = 0; k < d; k++) {
        double theta = -2 * M_PI * k / d;
        omega_inv(k) = std::polar(1.0, theta); // e^{i * theta}
    }
    
    Eigen::MatrixXcd v_omega_inv(d, d);
    
    for (int i = 0; i < d; i++) {
        std::complex<double> w = omega_inv(i);
        std::complex<double> val(1.0, 0.0);
        for (int j = 0; j < d; j++) {
            v_omega_inv(i, j) = val;
            val *= w;
        }
    }
    
    Eigen::MatrixXcd v_inv_omega_theo = v_omega_inv / d;
    Eigen::VectorXd inv_binom = inv_binom_coef(d - 1);
    Eigen::MatrixXcd res = (v_inv_omega_theo * inv_binom) / d;
    
    return res;
}

Eigen::MatrixXcd store_complex_v_invc(int d) {
    // Pre-store v_invc: v(z)^-1 @ c / d up to maximum tree depth where z are complex roots of unity
    Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(d + 1, d);
    
    for (int i = 1; i <= d; i++) {
        Eigen::MatrixXcd temp = complex_v_invc_degree(i);
        for (int j = 0; j < i; j++) {
            res(i, j) = temp(j, 0);
        }
    }
    
    return res;
}

Eigen::MatrixXcd store_complex_root(int d) {
    // Pre-store the complex root of unity z
    Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(d + 1, d);
    
    for (int i = 1; i <= d; i++) {
        for (int j = 0; j < i; j++) {
            double theta = 2 * M_PI * j / i;
            res(i, j) = std::polar(1.0, theta); // e^{i * theta}
        }
    }
    
    return res;
}

double complex_dot_v2(const Eigen::VectorXcd& p, const Eigen::VectorXcd& v_invc, int d) {
    // Return the dot product: C(z) * P(z) / d where z are the complex roots of unity
    // Using the fact that for roots of unity, elements come in complex conjugate pairs
    
    int len_p = p.size();
    std::complex<double> res = p(0) * v_invc(0);
    
    if (d % 2 == 0) {
        // For even d, handle the middle element separately
        for (int i = 1; i < len_p - 1; i++) {
            res += 2.0 * p(i) * v_invc(i);
        }
        res += p(len_p - 1) * v_invc(len_p - 1);
    } else {
        // For odd d, all conjugate pairs
        for (int i = 1; i < len_p; i++) {
            res += 2.0 * p(i) * v_invc(i);
        }
    }
    
    return res.real();
}

void traversal_weight(
        const Eigen::VectorXd& x,
        int v,
        Eigen::VectorXd& w,
        const Eigen::VectorXi& children_left,
        const Eigen::VectorXi& children_right,
        const Eigen::VectorXi& feature,
        const Eigen::VectorXd& threshold,
        const Eigen::VectorXd& sample_weight,
        const Eigen::VectorXi& leaf_ind,
        Eigen::MatrixXd& w_res,
        Eigen::MatrixXi& w_ind,
        int depth,
        Eigen::VectorXi& met_feature
) {
    int v_l = children_left(v);
    int v_r = children_right(v);
    
    if (v_l < 0) {
        // This is a leaf node
        // Find the right location in leaf_ind vector
        for (int i = 0; i < leaf_ind.size(); ++i) {
            if (leaf_ind(i) == v) {
                // Update w_res and w_ind for features encountered on the path
                for (int tmp_depth = 0; tmp_depth < depth; ++tmp_depth) {
                    int feat = met_feature(tmp_depth);
                    w_res(i, feat) = w(tmp_depth);
                    w_ind(i, feat) = 1;
                }
                break;
            }
        }
    } else {
        // This is a split node
        int split_feature = feature(v);
        double split_threshold = threshold(v);

        // Find the most recent occurrence of the split feature in our path
        int former_depth = -1;
        for (int d = depth - 1; d >= 0; --d) {
            if (met_feature(d) == split_feature) {
                former_depth = d;
                break;
            }
        }

        // If we haven't seen this feature before, set the weight to 1
        if (former_depth == -1) {
            former_depth = depth;
            w(depth) = 1;
        }

        // Copy met_feature to preserve state for recursive calls
        Eigen::VectorXi met_feature_copy = met_feature;
        met_feature_copy(depth) = split_feature;

        // Create a copy of the weights for the right branch
        Eigen::VectorXd w_r = w;

        // Update weights based on which branch the sample takes
        if (x(split_feature) <= split_threshold) {
            // Sample goes to left branch
            w(depth) = w(former_depth) * sample_weight(v_l);
            w_r(depth) = 0;
        } else {
            // Sample goes to right branch
            w_r(depth) = w_r(former_depth) * sample_weight(v_r);
            w(depth) = 0;
        }

        // Recursively process left and right children
        traversal_weight(x, v_l, w, children_left, children_right, feature, threshold, 
                        sample_weight, leaf_ind, w_res, w_ind, depth + 1, met_feature_copy);
        traversal_weight(x, v_r, w_r, children_left, children_right, feature, threshold, 
                        sample_weight, leaf_ind, w_res, w_ind, depth + 1, met_feature_copy);
    }
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> weight(
    const Eigen::VectorXd& x,
    const TreeSummary& summary_tree
) {
    int p = x.size();
    int d = summary_tree.max_depth;
    const Eigen::VectorXi& feature_uniq = summary_tree.feature_uniq;
    
    // Find all leaf nodes
    Eigen::VectorXi leaf_indices(0);
    for (int i = 0; i < summary_tree.node_count; ++i) {
        if (summary_tree.children_left(i) == -1) {
            // This is a leaf node
            Eigen::VectorXi new_leaf_indices(leaf_indices.size() + 1);
            if (leaf_indices.size() > 0) {
                new_leaf_indices.head(leaf_indices.size()) = leaf_indices;
            }
            new_leaf_indices(leaf_indices.size()) = i;
            leaf_indices = new_leaf_indices;
        }
    }
    
    // Initialize result matrices
    Eigen::MatrixXd w_res = Eigen::MatrixXd::Zero(leaf_indices.size(), p);
    Eigen::MatrixXi w_ind = Eigen::MatrixXi::Zero(leaf_indices.size(), p);
    
    // Initialize with default values
    for (int i = 0; i < leaf_indices.size(); ++i) {
        for (int j = 0; j < feature_uniq.size(); ++j) {
            w_res(i, feature_uniq(j)) = 1.0;
        }
    }
    
    // Initialize temporary weight vector and met_feature tracking
    Eigen::VectorXd w = Eigen::VectorXd::Zero(d);
    Eigen::VectorXi met_feature = Eigen::VectorXi::Constant(d, -1);
    
    // Begin traversal from the root
    traversal_weight(
        x, 0, w, 
        summary_tree.children_left, summary_tree.children_right,
        summary_tree.feature, summary_tree.threshold,
        summary_tree.sample_weight, leaf_indices,
        w_res, w_ind, 0, met_feature
    );
    
    return std::make_pair(w_res, w_ind);
}
