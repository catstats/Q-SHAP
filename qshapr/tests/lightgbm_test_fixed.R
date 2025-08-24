library(lightgbm)
library(qshapr)

# Load data
X <- as.matrix(read.csv("tests/X_data.csv"))
y <- as.matrix(read.csv("tests/y_data.csv"))

# Use very small subset for testing to avoid segfaults
X_small <- X[1:20, 1:3]
y_small <- y[1:20]

# Train LightGBM model with better parameters to ensure tree splits
dtrain <- lgb.Dataset(data = X_small, label = y_small)
model <- lgb.train(
  params = list(
    objective = "regression", 
    num_leaves = 7,  # Smaller trees 
    min_data_in_leaf = 2,  # Prevent overly complex trees
    feature_fraction = 1.0,
    lambda_l1 = 0,  
    lambda_l2 = 0,
    verbose = -1
  ),
  data = dtrain,
  nrounds = 2, # Fewer trees
  verbose = -1
)

# Create Q-SHAP explainer
explainer <- qshapr::create_tree_explainer(model)

# Calculate Q-SHAP R^2 contributions
rsq <- qshapr::qshap_rsq(explainer, X_small, y_small)

# Print results
print("Q-SHAP R^2 contributions:")
print(rsq)

rsq_sum <- sum(rsq)
print(paste("Sum of Q-SHAP R^2 contributions:", rsq_sum))

# Calculate true model R^2
ypred <- predict(model, X_small)
sst <- sum((y_small - mean(y_small))^2)
sse <- sum((y_small - ypred)^2)
true_rsq <- 1 - sse/sst
print(paste("True Model R^2:", true_rsq))

print(paste("Difference:", abs(rsq_sum - true_rsq)))
