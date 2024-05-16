from treeshap_rsq.utils import summarize_tree, simple_tree, tree_summary, weight, store_complex_root, store_complex_v_invc, xgb_formatter
from treeshap_rsq.treeshap_rsq import loss_treeshap
from types import SimpleNamespace

import sklearn
import numpy as np
import xgboost
import shap
import warnings
import json

class explainer:
    def __init__(self, model):

        implemented_model = (sklearn.tree.DecisionTreeRegressor,
                                sklearn.ensemble.GradientBoostingRegressor,
                                xgboost.sklearn.XGBRegressor)
        
        if isinstance(model, implemented_model):
            self.explainer = shap.TreeExplainer(model)
            if isinstance(model, (sklearn.tree.DecisionTreeRegressor,sklearn.ensemble.GradientBoostingRegressor)):
                self.max_depth = model.max_depth
            elif isinstance(model, xgboost.sklearn.XGBRegressor):
                # set to default value 6 if max_depth not set by user 
                self.max_depth = model.get_params().get("max_depth", 6)

            # store v_inc * c /d evaluated at complex roots     
            store_v_invc = store_complex_v_invc(self.max_depth * 2)
            store_z = store_complex_root(self.max_depth * 2)
        else:
            supported_models_str = ', '.join([m.__name__ for m in implemented_model])
            raise NotImplementedError(f"Model not supported yet. Supported models are: {supported_models_str}")
        

    def loss(self, x, y):
        """
        x: the original x
        y: the original y
        """
        max_depth = self.max_depth
        model = self.model
        store_v_invc = self.store_v_invc 
        store_z = self.store_z
        explainer = self.explainer

        sst = np.sum((y - np.mean(y)) ** 2)

        if isinstance(model, sklearn.tree.DecisionTreeRegressor):
            summary_tree = summarize_tree(model.tree_)
            loss = loss_treeshap(x, y, summary_tree, store_v_invc, store_z, explainer)

        # GBM 
        elif isinstance(model, sklearn.ensemble.GradientBoostingRegressor):
            ensemble_tree = model.estimators_
            num_tree = len(model)
            staged_predict = list(model.staged_predict(x))
            # learning_rate 
            alpha = model.learning_rate
            loss = np.zeros_like(x)
            
            for i in range(num_tree):
                if i==0:
                    res = y - np.mean(y)
                else:
                    res = y - staged_predict[i-1]
                    
                summary_tree = summarize_tree(ensemble_tree[i, 0].tree_)
                explainer = shap.TreeExplainer(ensemble_tree[i, 0])
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, alpha)

        # XGBOOST 
        elif isinstance(model, xgboost.sklearn.XGBRegressor):

            xgb_booster = model.get_booster()           

            model.save_model("xgb_model_tmp.json")
            with open('xgb_model_tmp.json', 'r') as file:
                model_data = json.load(file)

            xgb_res = xgb_formatter(model_data, max_depth)

            num_tree = len(xgb_res)

            warnings.filterwarnings("ignore", module="xgb")

            loss = np.zeros_like(x)

            for i in range(num_tree):
                # get summary_tree first 
                if i==0:
                    res = y - np.mean(y)
                else:
                    res = y - model.predict(x, iteration_range=(0, i))
                
                summary_tree = summarize_tree(xgb_res[i])
                explainer = shap.TreeExplainer(xgb_booster[i])
                
                # learning rate is different
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, 1)

        return loss
    

    def rsq(self, x, y, loss_out=False):
        """
        x: the original x
        y: the original y
        loss_out: output loss or not
        """
        explainer = self.explainer
        sst = np.sum((y - np.mean(y)) ** 2)

        loss = self.loss(x, y)
        rsq = 0 - np.sum(loss, axis=0)/sst

        if loss_out:
            return SimpleNamespace(rsq=rsq, loss=loss)
        else:
            return rsq
        