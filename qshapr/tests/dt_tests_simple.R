library(rpart)
library(fastshap)
library(qshapr)

set.seed(0)
n_samples <- 2000
max_depth <- 1
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

shap_vals <- fastshap::explain(
    object = tree_regressor,
    X = X_df,
    pred_wrapper = pred_fun,
    nsim = 100
)

rsq_res <- qshapr::rsq(explainer, X, y, shap_vals)

print(paste("QSHAP R^2 result: ", sum(rsq_res)))
print(rsq_res)