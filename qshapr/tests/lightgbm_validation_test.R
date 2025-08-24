library(lightgbm)
library(qshapr)

# Create synthetic data like the Python notebook  
set.seed(0)
n_samples <- 100
X <- matrix(runif(n_samples * 3), n_samples, 3)  # 3 features
y <- X[,1] + 2 * X[,2] + 0.5 * X[,3] + rnorm(n_samples, sd=0.1)

# Train LightGBM model (single tree like Python test)
dtrain <- lgb.Dataset(data = X, label = y)
model <- lgb.train(
  params = list(
    objective = "regression", 
    num_leaves = 15, 
    min_data_in_leaf = 5,
    verbose = -1
  ),
  data = dtrain,
  nrounds = 1,  # Single tree to match Python
  verbose = -1
)

cat("=== LightGBM Q-SHAP R² Validation ===\n")

# Calculate true model R²
ypred <- predict(model, X)
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
true_rsq <- 1 - sse/sst

cat("True Model R²:", true_rsq, "\n")

# Create Q-SHAP explainer and calculate R² contributions
explainer <- qshapr::create_tree_explainer(model)
rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)

cat("Q-SHAP R² contributions:", rsq_contributions, "\n")
cat("Sum of Q-SHAP R² contributions:", sum(rsq_contributions), "\n")
cat("Difference:", abs(sum(rsq_contributions) - true_rsq), "\n")

# Check if they match (like Python notebook)
tolerance <- 1e-10
match_result <- abs(sum(rsq_contributions) - true_rsq) < tolerance
cat("Q-SHAP R² sum equals Model R²?:", match_result, "\n")

if (match_result) {
  cat("SUCCESS: R² values match perfectly! 🎉\n")
} else {
  cat("ISSUE: R² values don't match. Debugging needed.\n")
  
  # Debug info
  cat("\nDEBUG INFO:\n")
  cat("Number of trees:", model$current_iter(), "\n")
  cat("Tree structure:\n")
  tree1 <- explainer$trees[[1]]
  cat("  Node count:", tree1$node_count, "\n")
  cat("  Features used:", unique(tree1$feature[tree1$feature >= 0]), "\n")
  
  # Test SHAP values directly
  cat("Testing LightGBM SHAP values:\n")
  shap_values <- predict(model, X[1:5,], type = "contrib")
  cat("  SHAP shape:", dim(shap_values), "\n")
  cat("  First sample SHAP:", shap_values[1,], "\n")
  cat("  First sample prediction:", predict(model, X[1,, drop=FALSE]), "\n")
  cat("  SHAP sum + bias:", sum(shap_values[1,]), "\n")
}
