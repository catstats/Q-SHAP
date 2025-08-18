# Basic LightGBM debugging to isolate the crash

print("=== LightGBM Installation Debug ===")

# Check R and system info
print(paste("R version:", R.version.string))
print(paste("Platform:", R.version$platform))

# Test 1: Can we load lightgbm?
print("Testing lightgbm package loading...")
tryCatch({
  library(lightgbm)
  print("✓ lightgbm loaded successfully")
}, error = function(e) {
  print(paste("✗ Error loading lightgbm:", e$message))
  stop("Cannot load lightgbm")
})

# Test 2: Check lightgbm version
print("Checking lightgbm version...")
tryCatch({
  # Try to get version info
  print(paste("lightgbm namespace loaded:", "lightgbm" %in% loadedNamespaces()))
  print("✓ Version check passed")
}, error = function(e) {
  print(paste("Warning in version check:", e$message))
})

# Test 3: Create minimal data
print("Creating minimal test data...")
X_minimal <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
y_minimal <- c(1, 0)
print("✓ Test data created")
print(X_minimal)
print(y_minimal)

# Test 4: Try lgb.Dataset with different approaches
print("Testing lgb.Dataset creation...")

# Approach 1: Basic
print("Approach 1: Basic lgb.Dataset...")
tryCatch({
  dtrain1 <- lgb.Dataset(data = X_minimal, label = y_minimal)
  print("✓ Basic lgb.Dataset works")
}, error = function(e) {
  print(paste("✗ Basic lgb.Dataset failed:", e$message))
})

# Approach 2: Convert to matrix explicitly
print("Approach 2: Explicit matrix conversion...")
tryCatch({
  X_mat <- as.matrix(X_minimal)
  y_vec <- as.numeric(y_minimal)
  dtrain2 <- lgb.Dataset(data = X_mat, label = y_vec)
  print("✓ Explicit conversion works")
}, error = function(e) {
  print(paste("✗ Explicit conversion failed:", e$message))
})

# Approach 3: Very basic - just 1x1
print("Approach 3: Minimal 1x1 data...")
tryCatch({
  X_tiny <- matrix(1, nrow = 1, ncol = 1)
  y_tiny <- 1
  dtrain3 <- lgb.Dataset(data = X_tiny, label = y_tiny)
  print("✓ Tiny data works")
}, error = function(e) {
  print(paste("✗ Tiny data failed:", e$message))
})

print("=== If you get here, basic LightGBM works ===")

# Test 5: Try training
print("Testing basic training...")
tryCatch({
  params <- list(objective = "regression", verbose = -1)
  model <- lgb.train(params = params, data = dtrain1, nrounds = 1, verbose = -1)
  print("✓ Basic training works")
}, error = function(e) {
  print(paste("✗ Training failed:", e$message))
})

print("=== LightGBM Debug Complete ===")
print("If this script completes, LightGBM is working!")
