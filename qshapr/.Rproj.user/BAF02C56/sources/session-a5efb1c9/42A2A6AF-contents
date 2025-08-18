#' @import Rcpp
#' @importFrom methods new
#' @useDynLib qshapr, .registration = TRUE
NULL

#' Create a QSHAPR Tree Explainer
#' 
#' Add this later
#' 
#' @param tree_model A tree model object
#' @return A TreeExplainer object
#' @export
create_tree_explainer <- function(tree_model, max_depth = NULL, base_score = NULL, ...) {
  UseMethod("create_tree_explainer")
}

#' @export
create_tree_explainer.xgb.Booster <- function(tree_model, ...) {

  # Apparently we don't have to do some weird JSON workaround
  max_depth <- if (!is.null(tree_model$params$max_depth)) tree_model$params$max_depth else 6
  base_score <- if (!is.null(tree_model$params$base_score)) tree_model$params$base_score else 0.5


  xgb_trees <- xgb_formatter(tree_model, max_depth)

  explainer <- structure(list(
    model = tree_model,
    model_type = "xgboost",
    max_depth = max_depth,
    base_score = base_score,
    trees = xgb_trees,
    store_v_invc = store_complex_v_invc(max_depth * 2),
    store_z = store_complex_root(max_depth * 2)
  ), class = c("qshapr_tree_explainer", "xgboost_explainer"))

  explainer
}

# #' @export
# create_tree_explainer.lgb.Booster <- function(tree_model, max_depth = NULL, ...) {
# }

# #' @export
# create_tree_explainer.gbm <- function(tree_model, max_depth = NULL, ...) {
# }

#' @export
create_tree_explainer.default <- function(tree_model, ...) {
  stop(sprintf("create_tree_explainer not implemented for class %s", class(tree_model)[1]))
}

#' Q-SHAP Loss
#' @export
qshap_loss <- function(explainer, x, y, y_mean_ori = NULL) {
  UseMethod("qshap_loss")
}

#' @export
qshap_loss.qshapr_tree_explainer <- function(explainer, x, y, y_mean_ori = NULL) {
  switch(explainer$model_type,
    "xgboost" = qshap_loss_xgboost(explainer, x, y, y_mean_ori),
    # "lightgbm" = qshap_loss_lightgbm(explainer, x, y, y_mean_ori, progress_bar),
    # "gbm" = qshap_loss_gbm(explainer, x, y, y_mean_ori, progress_bar),
    stop("Unknown model type: ", explainer$model_type)
  )
}

#' @export 
qshap_rsq <- function(explainer, x, y, loss_out = FALSE, nsample = NULL,
                      nfrac = NULL, random_state = 42) {
  # Sampling logic
  if (!is.null(nsample)) {
    if (nsample <= 0 || nsample >= nrow(x)) {
      stop("nsample must be > 0 and < total number of samples")
    }
    set.seed(random_state)
    sample_idx <- sample(nrow(x), nsample, replace = FALSE)
    x <- x[sample_idx, , drop = FALSE]
    y <- y[sample_idx]
  } else if (!is.null(nfrac)) {
    if (nfrac <= 0 || nfrac >= 1) {
      stop("nfrac must be between 0 and 1")
    }
    set.seed(random_state)
    nsample <- floor(nrow(x) * nfrac)
    sample_idx <- sample(nrow(x), nsample, replace = FALSE)
    x <- x[sample_idx, , drop = FALSE]
    y <- y[sample_idx]
  }

  y_mean_ori <- mean(y)
  sst <- sum((y - y_mean_ori)^2)

  # Calculate loss
  loss <- qshap_loss(explainer, x, y, y_mean_ori)

  # Calculate R-squared
  rsq <- -colSums(loss) / sst

  if (loss_out) {
    return(list(rsq = rsq, loss = loss))
  } else {
    return(rsq)
  }
}