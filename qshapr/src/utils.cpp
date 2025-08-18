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
    int len_p = p.size();
    std::complex<double> res = p(0) * v_invc(0);
    
    if (d % 2 == 0) {
        for (int i = 1; i < len_p - 1; i++) {
            res += 2.0 * p(i) * v_invc(i);
        }
        res += p(len_p - 1) * v_invc(len_p - 1);
    } else {
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
        for (int i = 0; i < leaf_ind.size(); ++i) {
            if (leaf_ind(i) == v) {
                for (int tmp_depth = 0; tmp_depth < depth; ++tmp_depth) {
                    int feat = met_feature(tmp_depth);
                    w_res(i, feat) = w(tmp_depth);
                    w_ind(i, feat) = 1;
                }
                break;
            }
        }
    } else {
        int split_feature = feature(v);
        double split_threshold = threshold(v);

        int former_depth = -1;
        for (int d = depth - 1; d >= 0; --d) {
            if (met_feature(d) == split_feature) {
                former_depth = d;
                break;
            }
        }

        if (former_depth == -1) {
            former_depth = depth;
            w(depth) = 1;
        }

        Eigen::VectorXi met_feature_copy = met_feature;
        met_feature_copy(depth) = split_feature;

        Eigen::VectorXd w_r = w;

        if (x(split_feature) <= split_threshold) {
            w(depth) = w(former_depth) * sample_weight(v_l);
            w_r(depth) = 0;
        } else {
            w_r(depth) = w_r(former_depth) * sample_weight(v_r);
            w(depth) = 0;
        }

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
    
    Eigen::VectorXi leaf_indices(0);
    for (int i = 0; i < summary_tree.node_count; ++i) {
        if (summary_tree.children_left(i) == -1) {
            Eigen::VectorXi new_leaf_indices(leaf_indices.size() + 1);
            if (leaf_indices.size() > 0) {
                new_leaf_indices.head(leaf_indices.size()) = leaf_indices;
            }
            new_leaf_indices(leaf_indices.size()) = i;
            leaf_indices = new_leaf_indices;
        }
    }
    
    Eigen::MatrixXd w_res = Eigen::MatrixXd::Zero(leaf_indices.size(), p);
    Eigen::MatrixXi w_ind = Eigen::MatrixXi::Zero(leaf_indices.size(), p);
    
    for (int i = 0; i < leaf_indices.size(); ++i) {
        for (int j = 0; j < feature_uniq.size(); ++j) {
            w_res(i, feature_uniq(j)) = 1.0;
        }
    }
    
    Eigen::VectorXd w = Eigen::VectorXd::Zero(d);
    Eigen::VectorXi met_feature = Eigen::VectorXi::Constant(d, -1);
    
    traversal_weight(
        x, 0, w, 
        summary_tree.children_left, summary_tree.children_right,
        summary_tree.feature, summary_tree.threshold,
        summary_tree.sample_weight, leaf_indices,
        w_res, w_ind, 0, met_feature
    );
    
    return std::make_pair(w_res, w_ind);
}
