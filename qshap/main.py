from qshap.utils import summarize_tree, simple_tree, tree_summary, weight, store_complex_root, store_complex_v_invc, xgb_formatter, lgb_formatter, lgb_shap, divide_chunks
from qshap.qshap import loss_treeshap

from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import sklearn
import numpy as np
import xgboost
import lightgbm
import shap
import warnings
import os
import json
import uuid

class gazer:
    def __init__(self, model):
        self.model = model
        implemented_model = (sklearn.tree.DecisionTreeRegressor,
                                sklearn.ensemble.GradientBoostingRegressor,
                                xgboost.sklearn.XGBRegressor,
                                lightgbm.sklearn.LGBMRegressor)
        
        if isinstance(model, implemented_model):
            self.explainer = shap.TreeExplainer(model)
            if isinstance(model, (sklearn.tree.DecisionTreeRegressor,sklearn.ensemble.GradientBoostingRegressor)):
                self.max_depth = model.max_depth
            elif isinstance(model, lightgbm.sklearn.LGBMRegressor):
                self.max_depth = model.get_params()['max_depth']
            elif isinstance(model, xgboost.sklearn.XGBRegressor):
                # set to default value 6 if max_depth not set by user 
                max_depth_xgb = model.get_params().get("max_depth")
                if max_depth_xgb is not None:
                    self.max_depth = max_depth_xgb
                else:
                    self.max_depth = 6

                unique_filename = str(uuid.uuid4())
                model_filename = f"xgb_model_{unique_filename}.json"
                model.save_model(model_filename)

               # Load the model data
                with open(model_filename, 'r') as file:
                    model_data = json.load(file)
                # Delete it after loading 
                os.remove(model_filename)
                self.base_score = np.float64(model_data["learner"]["learner_model_param"]['base_score'])
                self.xgb_res = xgb_formatter(model_data, self.max_depth)
                del model_data

            # store v_inc * c /d evaluated at complex roots     
            self.store_v_invc = store_complex_v_invc(self.max_depth * 2)
            self.store_z = store_complex_root(self.max_depth * 2)
        else:
            supported_models_str = ', '.join([m.__name__ for m in implemented_model])
            raise NotImplementedError(f"Model not supported yet. Supported models are: {supported_models_str}")
        

    def loss(self, x, y, y_mean_ori=None, progress_bar=True):
        """
        Parameters
        -x: x
        -y: y
        -y_mean_ori: mean of the original
        -progress_bar: whether show the progress bar or not
        """
        max_depth = self.max_depth
        model = self.model
        store_v_invc = self.store_v_invc 
        store_z = self.store_z
        explainer = self.explainer

        if y_mean_ori is None:
            y_mean_ori = np.mean(y)

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
        
            loss = np.zeros_like(x, dtype=np.float64)
            
            iterator = tqdm(range(num_tree)) if progress_bar else range(num_tree)

            for i in iterator:
                if i==0:
                    res = y - y_mean_ori
                else:
                    res = y - staged_predict[i-1]
                    
                summary_tree = summarize_tree(ensemble_tree[i, 0].tree_)
                explainer = shap.TreeExplainer(ensemble_tree[i, 0])
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, alpha)

        # XGBOOST 
        elif isinstance(model, xgboost.sklearn.XGBRegressor):

            xgb_booster = model.get_booster()           
            xgb_res = self.xgb_res
            num_tree = len(xgb_res)

            warnings.filterwarnings("ignore", module="xgb")

            loss = np.zeros_like(x, dtype=np.float64)
            
            iterator = tqdm(range(num_tree)) if progress_bar else range(num_tree)

            for i in iterator:
                # get summary_tree first 
                if i==0:
                    res = y - self.base_score
                else:
                    res = y - model.predict(x, iteration_range=(0, i))
                
                summary_tree = summarize_tree(xgb_res[i])
                explainer = shap.TreeExplainer(xgb_booster[i])
                
                # learning rate is different
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, 1)

        # LightGBM
        elif isinstance(model, lightgbm.sklearn.LGBMRegressor):
            lgb_res = lgb_formatter(model.booster_.trees_to_dataframe(), max_depth)
            lgb_shap_res = lgb_shap(lgb_res)
            num_tree = model.n_iter_

            loss = np.zeros_like(x, dtype=np.float64)

            iterator = tqdm(range(num_tree)) if progress_bar else range(num_tree)

            for i in iterator:
                # get summary_tree first 
                if i==0:
                    res = y 
                else:
                    res = y - model.predict(x, num_iteration=i, raw_score=True)
                
                summary_tree = summarize_tree(lgb_res[i])
                explainer = shap.TreeExplainer(lgb_shap_res[i])
                
                # learning rate is different
                loss += loss_treeshap(x, res, summary_tree, store_v_invc, store_z, explainer, 1)

        return loss
    

    def rsq(self, x, y, loss_out=False, ncore=1, nsample=None, nfrac=None, random_state=42, progress_bar=True):
        """
        Parameters
        -x: the original x
        -y: the original y
        -loss_out: output loss or not
        -nsample: number of samples to sample from, by default use all samples
        -nfrac: fraction of samples to sample from, by default 1, use all samples
        -ncore: number of cores to use, with default value 1. It will NOT be beneficial for small datasets and shallow depth.
        -random_state: control random seed for numpy
        -progress_bar: whether show the progress bar or not

        Return
        Shapley R-squared
        """ 

        
        if nsample is not None:
            if nsample <=0 or nsample >= x.shape[0]:
                raise ValueError("Samping sample size (nsample) must be larger than 0 and smaller than the total number of samples, use None for no sampling.")
            np.random.seed(random_state)
            sample_ind = np.random.choice(len(x), nsample, replace=False)
            x = x[sample_ind]
            y = y[sample_ind]
   
        if nfrac is not None and nsample is None:
            if nfrac <= 0 or nfrac >= 1:
                raise ValueError("Sample fraction (nfrac) must be between (0, 1), use None for no sampling.")
            np.random.seed(random_state)
            nsample = int(len(x) * nfrac)
            sample_ind = np.random.choice(len(x), nsample, replace=False)
            x = x[sample_ind]
            y = y[sample_ind]
                  
        max_core = os.cpu_count()
        if ncore == -1:
            ncore = os.cpu_count()
        ncore = min(max_core, ncore)

        explainer = self.explainer
        y_mean_ori = np.mean(y)
        sst = np.sum((y - y_mean_ori) ** 2)
        
        if ncore==1:
            loss = self.loss(x, y, y_mean_ori=y_mean_ori, progress_bar=progress_bar)
        else:
            x_chunks = divide_chunks(x, ncore)
            y_chunks = divide_chunks(y, ncore)

            with ProcessPoolExecutor(max_workers=ncore) as executor:
                # Submit all chunks for processing
                futures = [executor.submit(self.loss, x_chunks[i], y_chunks[i], y_mean_ori, False) for i in range(ncore)]

                # Assign progress bar to a variable
                iterator = tqdm(futures, desc="Processing", total=len(futures)) if progress_bar else futures

                # Wait for all futures to complete and collect results
                results = [future.result() for future in iterator]
            
            loss = np.concatenate(results) 

        rsq = 0 - np.sum(loss, axis=0)/sst

        if loss_out:
            return SimpleNamespace(rsq=rsq, loss=loss)
        else:
            return rsq
        

    def gcorr(rsq_res):
        """
        Parameters
        -rsq_res: the rsq result from calling gazer.rsq

        Return
        Generalized correlation (Square root of Shapley R-squared)
        """
        res = np.sqrt(rsq_res)
        return rsq_res