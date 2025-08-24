library(lightgbm)
library(qshapr)

# Create synthetic data  
set.seed(0)
n_samples <- 100
X <- matrix(runif(n_samples * 3), n_samples, 3)
y <- X[,1] + 2 * X[,2] + 0.5 * X[,3] + rnorm(n_samples, sd=0.1)

# Train LightGBM model
dtrain <- lgb.Dataset(data = X, label = y)
model <- lgb.train(
  params = list(objective = "regression", num_leaves = 15, min_data_in_leaf = 5, verbose = -1),
  data = dtrain,
  nrounds = 1,
  verbose = -1
)

# Calculate true model R²
ypred <- predict(model, X)
true_rsq <- 1 - sum((y - ypred)^2) / sum((y - mean(y))^2)

# Calculate Q-SHAP R² contributions
explainer <- qshapr::create_tree_explainer(model)
rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)

# Show the two numbers that match
print(paste("Q-SHAP R² sum:", sum(rsq_contributions)))
print(paste("True Model R²:", true_rsq))