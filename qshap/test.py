from qshap import gazer
import sklearn.ensemble
import numpy as np
import time as time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Binarizer
import xgboost
import lightgbm

from sklearn.datasets import make_regression

# Install the developer's version
# python -m pip install .
# If no priviledge 
# sudo -H python -m pip install .

# Let's start with a toy example with 1000 samples and 1000 features
np.random.seed(0)
x, y, coefficients = make_regression(n_samples=1000, n_features=1000, n_informative=5, coef=True, random_state=0)
#binarizer = Binarizer(threshold=np.median(x))
#x_binary = binarizer.transform(x)

# model fitting
# scikit learn decision tree example, the initial run after import will be slower since numba is comiling
max_depth = 2
tree_regressor = DecisionTreeRegressor(max_depth=max_depth)
tree_fit = tree_regressor.fit(x, y)

start = time.time()
gazer_rsq = gazer(tree_regressor)
rsq_res = gazer.rsq(gazer_rsq, x, y)
end = time.time()
print("time: " + str(end - start))

# Let's check the real R^2
ypred = tree_regressor.predict(x)
sst = np.sum((y - np.mean(y)) ** 2)
sse = np.sum((y - ypred) ** 2)
model_rsq = 1 - sse/sst

print("Treeshap R^2 sum is: " + str(np.sum(rsq_res)))
print("Model R^2 is: " + str(model_rsq) + "\n")
#print("Treeshap R^2 sum and the model R^2 is equal?:  " + str(round(np.sum(rsq_res), 3)==round(model_rsq, 3)) + "!!!!!")

# scikit learn gbdt example. The usage is the same.
# model fitting
max_depth = 2
n_estimators = 50
tree_regressor = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
tree_fit = tree_regressor.fit(x, y)

# start to explain 
start = time.time()
gazer_rsq = gazer(tree_regressor)
rsq_res = gazer.rsq(gazer_rsq, x, y)
end = time.time()
print("time: " + str(end - start))

# Let's check the real R^2
ypred = tree_regressor.predict(x)
sst = np.sum((y - np.mean(y)) ** 2)
sse = np.sum((y - ypred) ** 2)
model_rsq = 1 - sse/sst

print("Treeshap R^2 sum is: " + str(np.sum(rsq_res)))
print("Model R^2 is: " + str(model_rsq) + "\n")

# scikit learn xgboost example. The usage is the same again.
# model fitting
max_depth = 2
n_estimators = 50
tree_regressor = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
tree_regressor.fit(x, y)


# start to explain 
start = time.time()
gazer_rsq = gazer(tree_regressor)
rsq_res = gazer.rsq(gazer_rsq, x, y)
end = time.time()
print("time: " + str(end - start))

# Let's check the real R^2
ypred = tree_regressor.predict(x)
sst = np.sum((y - np.mean(y)) ** 2)
sse = np.sum((y - ypred) ** 2)
model_rsq = 1 - sse/sst

print("Treeshap R^2 sum is: " + str(np.sum(rsq_res)))
print("Model R^2 is: " + str(model_rsq) + "\n")

np.sum(gazer_rsq.explainer.shap_values(x)*y[:, np.newaxis]/sst, axis=0)

# scikit learn lightGBM example. The usage is the same again.
# model fitting
max_depth = 2
n_estimators = 50
tree_regressor = lightgbm.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, verbose=-1)
tree_regressor.fit(x, y)

# start to explain 
start = time.time()
gazer_rsq = gazer(tree_regressor)
rsq_res = gazer.rsq(gazer_rsq, x, y)
end = time.time()
print("time: " + str(end - start))

# Let's check the real R^2
ypred = tree_regressor.predict(x)
sst = np.sum((y - np.mean(y)) ** 2)
sse = np.sum((y - ypred) ** 2)
model_rsq = 1 - sse/sst

print("Treeshap R^2 sum is: " + str(np.sum(rsq_res)))
print("Model R^2 is: " + str(model_rsq) + "\n")

# if you would like to use sampling
start = time.time()
gazer_rsq = gazer(tree_regressor)
rsq_res_sample = gazer.rsq(gazer_rsq, x, y, nsample=128)
# rsq_res_frac = gazer.rsq(gazer_rsq, x, y, nfrac=0.5)
end = time.time()
print("time: " + str(end - start))

# Let's check the real R^2
ypred = tree_regressor.predict(x)
sst = np.sum((y - np.mean(y)) ** 2)
sse = np.sum((y - ypred) ** 2)
model_rsq = 1 - sse/sst

print("Treeshap R^2 sum is: " + str(np.sum(rsq_res_sample)))
print("Model R^2 is: " + str(model_rsq) + "\n")


# output generalized correlation (Square root of Shapley R squared)
gcorr_res = gazer.gcorr(rsq_res)

# extract loss example 

# Or simply output both 
rsq_res2 = gazer.rsq(gazer_rsq, x, y, loss_out=True)
rsq_res2.rsq
rsq_res2.loss

# You can extract loss only by, it decompose the loss for each sample
# But directly calling loss doesn't support parallel computing yet
loss_res = gazer.loss(gazer_rsq, x, y)

# Interstingly, you can calculate loss decomposition for arbitrarily sample (which doesn't make sense for R^2)
# This will be particular fast if you only want to several samples among a large dataset
# All you have to do is add the original mean of y
loss_sample1 = gazer.loss(gazer_rsq, x[0:1, :], y[0], y_mean_ori=np.mean(y))
loss_multiple = gazer.loss(gazer_rsq, x[0:99], y[0:99], y_mean_ori=np.mean(y))
# Check if the result match
#print(np.sum(np.abs(loss_res[0] - loss_sample1)))
#print(np.sum(np.abs(loss_res[0:99] - loss_multiple)))

# Visualize rsq 
# First we can import the module
from qshap import vis

# # Generate feature names using list comprehension and format them
feature_names = np.array([f"feature{i}" for i in range(1, rsq_res.shape[0]+1)])

# # Give it a name and rotate
vis.rsq(rsq_res, color_map_name="Pastel2", label=feature_names, rotation=45)

# default
vis.rsq(rsq_res)

# plot up to a certain cutoff
vis.rsq(rsq_res, cutoff=0.1, model_rsq=False)

# change color 
vis.rsq(rsq_res, color_map_name="Pastel2")
# Give a horizontal plot, hide model rsq, change the number of features to show, and you can always save it
vis.rsq(rsq_res, color_map_name="PuBu", horizontal=True, model_rsq=False, max_feature=15, save_name="rsq_eg")

# Elbow plot 
vis.elbow(rsq_res, max_comp=15)

vis.cumu(rsq_res)

# Vidualize individual loss
# The interactive effect works with jupyter notebook
vis.loss(loss_res)

# Find a lovely plot for one sample and save it, say for the 10-th sample
vis.loss(loss_res, save_ind=10)

# Visualize generalized rsq
vis.gcorr(rsq_res)
