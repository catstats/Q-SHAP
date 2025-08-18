# Simple LightGBM verification test for Q-SHAP R package
# Follows the same pattern as XGBoost test to verify correctness

library(lightgbm)

# Create synthetic regression data (equivalent to make_regression in Python)
set.seed(0)  # For reproducibility
n <- 5000
n_features <- 100
n_informative <- 5

# Generate random features
X <- matrix(rnorm(n * n_features), nrow = n, ncol = n_features)

# Create coefficients (only first n_informative features matter)
coefficients <- c(rnorm(n_informative, mean = 0, sd = 10), rep(0, n_features - n_informative))

# Generate target variable
y <- X %*% coefficients + rnorm(n, mean = 0, sd = 1)

# Model parameters
max_depth <- 1
learning_rate <- 0.05
n_estimators <- 1

# Create LightGBM dataset
dtrain <- lgb.Dataset(data = X, label = y)

# Define parameters for LightGBM regression
params <- list(
  objective = "regression",
  metric = "rmse",
  boosting_type = "gbdt",
  num_leaves = 2^max_depth,  # For max_depth=1, num_leaves=2
  learning_rate = learning_rate,
  lambda_l1 = 0,  # equivalent to alpha=0 in XGBoost
  lambda_l2 = 0,  # equivalent to reg_lambda=0 in XGBoost
  verbose = -1
)

# Train LightGBM model
print("Training LightGBM model...")
lgb_model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = n_estimators,
  verbose = -1
)

print("Finish")

# Get model predictions
ypred <- predict(lgb_model, X)

# Calculate model R-squared
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
model_rsq <- 1 - sse/sst

# Create Q-SHAP explainer
explainer <- create_tree_explainer(lgb_model)

# Time the Q-SHAP calculation
start_time <- Sys.time()

# Calculate Q-SHAP R-squared
qshap_result <- qshap_rsq(explainer, X, y, loss_out = TRUE)
rsq_res <- qshap_result$rsq

end_time <- Sys.time()
elapsed_time <- as.numeric(end_time - start_time, units = "secs")

print(paste("Elapsed time:", round(elapsed_time, 4), "seconds"))

# Print results
print("Q-SHAP R-squared values:")
print(rsq_res)
print(paste("Q-SHAP total R-squared:", round(sum(rsq_res), 6)))
print(paste("Model R-squared:", round(model_rsq, 6)))

# Verify they are approximately equal
difference <- abs(sum(rsq_res) - model_rsq)
print(paste("Difference:", round(difference, 8)))

if (difference < 1e-6) {
  print("✓ TEST PASSED: Q-SHAP and model R-squared match!")
} else if (difference < 1e-3) {
  print("⚠ TEST WARNING: Small difference between Q-SHAP and model R-squared")
} else {
  print("✗ TEST FAILED: Large difference between Q-SHAP and model R-squared")
}
