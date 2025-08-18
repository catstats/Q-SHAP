
#' Simple Tree Structure
simple_tree <- function(children_left,
                        children_right,
                        feature,
                        threshold,
                        max_depth,
                        n_node_samples,
                        value,
                        node_count) {
  structure(list(
    children_left   = children_left,
    children_right  = children_right,
    feature         = feature,
    threshold       = threshold,
    max_depth       = max_depth,
    n_node_samples  = n_node_samples,
    value           = value,
    node_count      = node_count
  ), class = "simple_tree")
}

#' Tree Summary Structure
tree_summary <- function(children_left,
                         children_right,
                         feature,
                         feature_uniq,
                         threshold,
                         max_depth,
                         sample_weight,
                         init_prediction,
                         node_count) {
  structure(list(
    children_left   = children_left,
    children_right  = children_right,
    feature         = feature,
    feature_uniq    = feature_uniq,
    threshold       = threshold,
    max_depth       = max_depth,
    sample_weight   = sample_weight,
    init_prediction = init_prediction,
    node_count      = node_count
  ), class = "tree_summary")
}

# Summarize tree function - converts simple_tree to tree_summary
#' @keywords internal
summarize_tree <- function(simple_tree_obj) {
  # Extract components from simple_tree object
  children_left <- simple_tree_obj$children_left
  children_right <- simple_tree_obj$children_right
  feature <- simple_tree_obj$feature
  threshold <- simple_tree_obj$threshold
  max_depth <- simple_tree_obj$max_depth
  n_node_samples <- simple_tree_obj$n_node_samples
  value <- simple_tree_obj$value
  node_count <- simple_tree_obj$node_count

  # Initialize sample_weight and init_prediction
  sample_weight <- rep(1.0, node_count)
  init_prediction <- rep(0.0, node_count)

  # Get root sample count
  n_root <- n_node_samples[1]  # R uses 1-based indexing

  # Recursive function to traverse and summarize tree
  traversal_summarize_tree <- function(v) {
    # Convert to 0-based indexing for comparison with -1
    v_l <- children_left[v]
    v_r <- children_right[v]

    n_v <- n_node_samples[v]

    if (v_l == -1) {  # leaf node
      init_prediction[v] <<- value[v] * n_v / n_root
    } else {
      # Convert back to 1-based indexing for R
      v_l_r <- v_l + 1
      v_r_r <- v_r + 1

      n_l <- n_node_samples[v_l_r]
      n_r <- n_node_samples[v_r_r]

      sample_weight[v_l_r] <<- n_v / n_l
      sample_weight[v_r_r] <<- n_v / n_r

      traversal_summarize_tree(v_l_r)
      traversal_summarize_tree(v_r_r)
    }
  }
  
  # Start traversal from root (index 1 in R)
  traversal_summarize_tree(1)
  
  # Get unique features (features >= 0)
  feature_uniq <- unique(feature[feature >= 0])
  feature_uniq <- sort(feature_uniq)
  
  # Create tree_summary object
  tree_summary(
    children_left = children_left,
    children_right = children_right,
    feature = feature,
    feature_uniq = feature_uniq,
    threshold = threshold,
    max_depth = max_depth,
    sample_weight = sample_weight,
    init_prediction = init_prediction,
    node_count = node_count
  )
}