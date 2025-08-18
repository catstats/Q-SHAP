# Step-by-step debug test to find where the crash occurs
library(lightgbm)
library(qshapr)

print("=== STEP 1: Creating minimal data ===")
set.seed(123)
n <- 10  # Very small
n_features <- 3
X <- matrix(rnorm(n * n_features), nrow = n, ncol = n_features)
y <- X[,1] + rnorm(n, 0, 0.1)

print(paste("Data created: X =", paste(dim(X), collapse="x"), ", y length =", length(y)))

print("=== STEP 2: Training minimal LightGBM model ===")
dtrain <- lgb.Dataset(data = X, label = y)
params <- list(objective = "regression", verbose = -1)
lgb_model <- lgb.train(params = params, data = dtrain, nrounds = 1, verbose = -1)
print("Model trained successfully")

print("=== STEP 3: Creating explainer ===")
explainer <- create_tree_explainer(lgb_model)
print("Explainer created successfully")
print(paste("Model type:", explainer$model_type))
print(paste("Trees count:", length(explainer$trees)))

print("=== STEP 4: Inspecting first tree structure ===")
first_tree <- explainer$trees[[1]]
print("First tree components:")
print(paste("Node count:", first_tree$node_count))
print(paste("Children left:", paste(first_tree$children_left, collapse=",")))
print(paste("Children right:", paste(first_tree$children_right, collapse=",")))
print(paste("Features:", paste(first_tree$feature, collapse=",")))

print("=== STEP 5: Testing summarize_tree function ===")
tryCatch({
  summary_tree <- summarize_tree(first_tree)
  print("summarize_tree completed successfully")
  print(paste("Summary tree node count:", summary_tree$node_count))
}, error = function(e) {
  print(paste("ERROR in summarize_tree:", e$message))
  print("This is likely where the crash occurs!")
  stop("summarize_tree failed")
})

print("=== STEP 6: Testing tiny qshap_loss call ===")
tryCatch({
  # Use just 1 sample and 3 features
  X_tiny <- X[1:1, , drop = FALSE]
  y_tiny <- y[1:1]
  
  print("Calling qshap_loss with tiny data...")
  loss <- qshap_loss(explainer, X_tiny, y_tiny)
  print("qshap_loss completed successfully!")
  print(paste("Loss dimensions:", paste(dim(loss), collapse="x")))
  
}, error = function(e) {
  print(paste("ERROR in qshap_loss:", e$message))
  print("qshap_loss is where the crash occurs!")
  stop("qshap_loss failed")
})

print("=== STEP 7: Testing qshap_rsq ===")
tryCatch({
  X_tiny <- X[1:2, , drop = FALSE]
  y_tiny <- y[1:2]
  
  print("Calling qshap_rsq with tiny data...")
  rsq_result <- qshap_rsq(explainer, X_tiny, y_tiny)
  print("qshap_rsq completed successfully!")
  print(paste("R-squared values:", paste(round(rsq_result, 4), collapse=",")))
  
}, error = function(e) {
  print(paste("ERROR in qshap_rsq:", e$message))
  print("qshap_rsq is where the crash occurs!")
  stop("qshap_rsq failed")
})

print("=== ALL STEPS COMPLETED SUCCESSFULLY! ===")
print("LightGBM Q-SHAP integration is working.")
