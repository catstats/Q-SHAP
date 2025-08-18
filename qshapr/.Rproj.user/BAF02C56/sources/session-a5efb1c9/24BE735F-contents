library(xgboost)
library(qshapr)
library(DiagrammeR)

url <- "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
housing <- read.csv(url)

housing$AveRooms <- housing$total_rooms / housing$households
housing$AveBedrms <- housing$total_bedrooms / housing$households
housing$AveOccup <- housing$population / housing$households

X <- data.frame(
    MedInc = housing$median_income,
    AveOccup = housing$AveOccup,
    Longitude = housing$longitude,
    Latitude = housing$latitude,
    HouseAge = housing$housing_median_age,
    AveRooms = housing$AveRooms,
    AveBedrms = housing$AveBedrms,
    Population = housing$population
)

y <- housing$median_house_value

# smallX = X[1:1000, ]
# smally = log(y[1:1000])

randomindices <- sample(nrow(X), 1000)

smallX = X[randomindices, ]
smally = log(y[randomindices])

model <- xgboost(
  data = as.matrix(smallX),
  label = smally,
  nrounds = 1,
  max_depth = 6,
  verbose = 0,
)

explainer <- qshapr::create_tree_explainer(model)
rsq = qshapr::qshap_rsq(explainer, smallX, smally)

print(rsq)

rsq_sum <- sum(rsq)

print(rsq_sum)

ypred = predict(model, as.matrix(smallX))
print(smally)
print(ypred)
print(unique(ypred))

plot(smally, ypred)

print(sum((smally - ypred)^2))
print(sum((smally - mean(smally))^2))

true_sum <- 1 - sum((smally-ypred)^2)/sum((smally-mean(smally))^2)
print(true_sum)

# xgb.plot.tree(model = model, trees = 1)