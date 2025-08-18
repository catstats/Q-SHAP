library(rpart)
library(fastshap)
library(qshapr)

set.seed(0)
n_samples <- 2000
max_depth <- 6
X <- matrix(runif(n_samples * 3), ncol = 3)
colnames(X) <- c("feature1", "feature2", "feature3")

y <- X[, 1] + 2 * X[, 2] + 0.5 * X[, 3] + rnorm(n_samples)

train_indices <- sample(1:n_samples, size = 0.8 * n_samples)
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

tree_regressor <- rpart(
  y_train ~ ., 
  data = data.frame(X_train, y_train), 
  method = "anova",
  control = rpart.control(maxdepth = max_depth)
)

# Create explainer
explainer <- qshapr::create_tree_explainer(tree_regressor)

print(explainer)

summary <- qshapr::get_summary(explainer)

print(summary)

# Calculate model R^2 directly 
X_df <- as.data.frame(X)
ypred <- predict(tree_regressor, newdata = X_df)
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
model_rsq <- 1 - sse / sst

print(paste("Model R^2 is: ", model_rsq))

# Now use TreeExplainer to compute R^2
pred_fun <- function(object, newdata) {
    predict(object, newdata = newdata)
}

# Use the same data frame format for consistency
shap_vals <- fastshap::explain(
    object = tree_regressor,
    X = X_df,
    pred_wrapper = pred_fun,
    nsim = 1000
)

# Add a manual verification that shap values sum to predictions minus baseline
baseline <- mean(ypred)
shap_sums <- rowSums(shap_vals)
pred_diffs <- ypred - baseline

# Calculate correlation between sum of shap values and model prediction differences
shap_pred_cor <- cor(shap_sums, pred_diffs)
print(paste("Correlation between sum of SHAP values and predictions:", shap_pred_cor))

# Calculate average absolute difference between shap sums and prediction differences
shap_pred_diff <- mean(abs(shap_sums - pred_diffs))
print(paste("Mean absolute difference between SHAP sums and prediction differences:", shap_pred_diff))

rsq_res <- qshapr::rsq(explainer, X, y, shap_vals)
print(paste("QSHAP R^2 result: ", sum(rsq_res)))

# Manual R^2 decomposition verification
# In theory, we should be able to compute feature-wise R^2 values
manual_rsq_values <- numeric(ncol(X))
for (j in 1:ncol(X)) {
  # For each feature, compute the reduction in SSE if we add this feature's SHAP values
  feature_contribution <- shap_vals[, j]
  # Multiply by feature contribution and sum
  manual_rsq_values[j] <- sum(2 * y * feature_contribution - feature_contribution^2) / sst
}

print(paste("Manual feature-wise R^2 sum:", sum(manual_rsq_values)))
print("Feature R^2 contributions:")
for (j in 1:ncol(X)) {
  print(paste(colnames(X)[j], ":", manual_rsq_values[j]))
}

# Theoretical verification using the loss formula from loss_treeshap
# The loss formula in loss_treeshap is: square_treeshap_x - 2 * (y * T0_x.T).T
# where square_treeshap_x corresponds to T2 values in the paper

# Compute each component of the loss manually
# Term 1: square_treeshap_x (approximated here using shap_vals^2)
square_terms <- shap_vals^2

# Term 2: 2 * (y * T0_x.T).T (which is 2 * y_i * shap_ij for each feature j and sample i)
cross_terms <- 2 * outer(y, rep(1, ncol(shap_vals))) * shap_vals

# Compute loss as: square_terms - cross_terms
manual_loss <- square_terms - cross_terms

# Sum the losses by column and divide by SST to get R^2 contributions
manual_rsq_from_loss <- -colSums(manual_loss) / sst

print(paste("Manual R^2 from loss formula:", sum(manual_rsq_from_loss)))
print("Feature R^2 contributions from loss formula:")
for (j in 1:ncol(X)) {
  print(paste(colnames(X)[j], ":", manual_rsq_from_loss[j]))
}