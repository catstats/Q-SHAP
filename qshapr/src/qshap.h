#ifndef QSHAP
#define QSHAP

#include <RcppEigen.h>
#include <Rcpp.h>
#include <complex>
#include <vector>

#include "utils.h"

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Eigen::MatrixXd T2(
    const Eigen::MatrixXd& x, 
    const Rcpp::List& tree_summary,
    const Eigen::MatrixXcd& store_v_invc, 
    const Eigen::MatrixXcd& store_z,
    bool parallel = true
);

void T2_sample(
    int i, 
    const Eigen::MatrixXd& w_matrix, 
    const Eigen::MatrixXi& w_ind, 
    const Eigen::VectorXd& init_prediction, 
    const Eigen::MatrixXcd& store_v_invc, 
    const Eigen::MatrixXcd& store_z, 
    Eigen::MatrixXd& shap_value, 
    const Eigen::VectorXi& feature_uniq
);

// [[Rcpp::export]]
Eigen::MatrixXd loss_treeshap(
        const Eigen::MatrixXd& x,
        const Eigen::VectorXd& y,
        const Rcpp::List& tree_summary,
        const Eigen::MatrixXcd& store_v_invc,
        const Eigen::MatrixXcd& store_z,
        const Eigen::MatrixXd& T0_x,
        double learning_rate = 1.0
);
    
#endif