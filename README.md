## Q-SHAP: Feature-Specific $R^2$ Values for Tree Ensembles
This package is used to compute feature-specific $R^2$ values, following Shapley decomposition of the total $R^2$, for tree ensembles in polynomial time.

This version only takes output from **XGBoost**, **scikit-learn Decision Tree**, and **scikit-learn GBDT**. We are working to update it for random forest in the next version. Please check **Q-SHAP Tutorial.ipynb** for more usage.

### Citation
```bibtex
@article{jiang2024feature,
  title={Feature-Specific Coefficients of Determination in Tree Ensembles},
  author={Jiang, Zhongli and Zhang, Dabao and Zhang, Min},
  journal={arXiv preprint arXiv:2407.03515},
  year={2024}
}
```


### References
- Jiang, Z., Zhang, D., & Zhang, M. (2024). Feature-Specific Coefficients of Determination in Tree Ensembles. arXiv preprint arXiv:2407.03515.
- Lundberg, Scott M., et al. "From local explanations to global understanding with explainable AI for trees." Nature machine intelligence 2.1 (2020): 56-67.
- Karczmarz, Adam, et al. "Improved feature importance computation for tree models based on the Banzhaf value." Uncertainty in Artificial Intelligence. PMLR, 2022.
- Bifet, Albert, Jesse Read, and Chao Xu. "Linear tree shap." Advances in Neural Information Processing Systems 35 (2022): 25818-25828.
- Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 2016.



### Task List

- [ ] Task 1: Completed task
- [ ] Task 2: Task in progress
- [ ] Task 3: Pending task
