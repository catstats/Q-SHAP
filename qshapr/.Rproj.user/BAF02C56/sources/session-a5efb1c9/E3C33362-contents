library(xgboost)
library(qshapr)

X <- as.matrix(read.csv("tests/X_data.csv"))
y <- as.matrix(read.csv("tests/y_data.csv"))

# smallX = X[1:1000, ]
# smally = log(y[1:1000])

# randomindices <- sample(nrow(X), 1000)

model <- xgboost(
  data = as.matrix(X),
  label = as.matrix(y),
  nrounds = 50,
  max_depth = 2,
  verbose = 0,
)

explainer <- qshapr::create_tree_explainer(model)
rsq = qshapr::qshap_rsq(explainer, X, y)

print(rsq)

rsq_sum <- sum(rsq)

print(rsq_sum)

ypred = predict(model, as.matrix(X))
# print(y)
# print(ypred)
# print(unique(ypred))

plot(y, ypred)

print(sum((y - ypred)^2))
print(sum((y - mean(y))^2))

true_sum <- 1 - sum((y-ypred)^2)/sum((y-mean(y))^2)
print(true_sum)
