# Minimal LightGBM test to debug the crash issue

library(lightgbm)

# Start with very small data to avoid memory issues
set.seed(123)
n <- 50  # Much smaller dataset
n_features <- 5

# Generate simple test data
X <- matrix(rnorm(n * n_features), nrow = n, ncol = n_features)
y <- X[,1] + 0.5 * X[,2] + rnorm(n, 0, 0.1)

print("Data created successfully")
print(paste("X dimensions:", paste(dim(X), collapse = " x ")))
print(paste("y length:", length(y)))

# Try basic LightGBM functionality first
print("Creating LightGBM dataset...")
dtrain <- lgb.Dataset(data = X, label = y)
print("Dataset created successfully")

# Very simple parameters
params <- list(
  objective = "regression",
  metric = "rmse",
  verbose = -1
)

print("Training LightGBM model...")
lgb_model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 1,  # Just 1 round to minimize complexity
  verbose = -1
)
print("Model trained successfully")

# Test prediction on a very small subset first
print("Testing prediction on small subset...")
X_small <- X[1:5, , drop = FALSE]
ypred_small <- predict(lgb_model, X_small)
print("Small prediction successful")
print(paste("Small predictions:", paste(round(ypred_small, 4), collapse = ", ")))

# If small prediction works, try larger
print("Testing prediction on full dataset...")
tryCatch({
  ypred <- predict(lgb_model, X)
  print("Full prediction successful")
  print(paste("First 5 predictions:", paste(round(ypred[1:5], 4), collapse = ", ")))
}, error = function(e) {
  print(paste("Error in prediction:", e$message))
  stop("Prediction failed")
})

# If we get here, basic LightGBM is working
print("Basic LightGBM functionality verified!")

# Now test our Q-SHAP functions
print("Testing Q-SHAP explainer creation...")
tryCatch({
  explainer <- create_tree_explainer(lgb_model)
  print("Explainer created successfully")
  print(paste("Explainer model type:", explainer$model_type))
}, error = function(e) {
  print(paste("Error creating explainer:", e$message))
  print(traceback())
  stop("Explainer creation failed")
})

print("All tests passed! LightGBM integration is working.")
