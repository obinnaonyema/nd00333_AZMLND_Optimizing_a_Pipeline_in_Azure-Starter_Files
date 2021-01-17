# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree.

In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

## Summary

The data set contains data about customers with which we seek to predict their interests. It contains customer data from a bank's marketing campaigns. [View data source from Kaggle here](https://www.kaggle.com/yufengsui/portuguese-bank-marketing-data-set?select=bank-full.csv). There are 21 columns and over 32000 rows containing data on employment, marital status, education, housing among others. Some of the columns will need to be one-hot encoded to have numeric values for them.

The best algorithm was the voting ensemble algorithm with 0.9181 accuracy derived via AutoML run.

## Scikit-learn Pipeline

a. Architecture: VM General Purpose CPU Cluster Compute D-Series V2

b. Data: CSV data source. Data is loaded and cleaned using the train.py script

c. Classification done via Scikit-learn Logistic Regression Model 

d. Hyperparameters: “C” is the regularization parameter, “max-iter” is the maximum number of iterations allowed

**What are the benefits of the parameter sampler you chose?**

Random Sampling allowed us optimize resource usage. It yields reasonably accurate results as other sampling methods such as Bayesian sampling, which would require higher computing power, longer run time and effectively a higher budget.

**What are the benefits of the early stopping policy you chose?**

BanditPolicy prevented unnecessary runs where there was no improvement in results after a few consecutive runs as defined. It allows for cost savings, even more than Median stopping policy, which compares performance against the median performance of previous runs.

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

The stopping policy here terminates a run if the primary metric value is outside 0.1 permitted slack of the primary metric value from best performing run so far.

## AutoML

Multiple models such as XGBoost and LightGBM were generated during the AutoML run but Voting Ensemble provided the best result.

![AutoML_run](https://github.com/obinnaonyema/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/autoML_run.PNG)

Below are the parameters for the component algorithms in the Voting Ensemble.
```
{'47': Pipeline(memory=None,
         steps=[('standardscalerwrapper',
                 <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f12520a9438>),
                ('xgboostclassifier',
                 XGBoostClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=0.8, eta=0.3, gamma=0,
                                   learning_rate=0.1, max_delta_step=0,
                                   max_depth=6, max_leaves=31,
                                   min_child_weight=1, missing=nan,
                                   n_estimators=100, n_jobs=-1, nthread=None,
                                   objective='reg:logistic', random_state=0,
                                   reg_alpha=2.5, reg_lambda=1.0416666666666667,
                                   scale_pos_weight=1, seed=None, silent=None,
                                   subsample=0.8, tree_method='auto',
                                   verbose=-10, verbosity=0))],
         verbose=False), '21': Pipeline(memory=None,
         steps=[('sparsenormalizer',
                 <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f12520aebe0>),
                ('xgboostclassifier',
                 XGBoostClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=0.7, eta=0.3, gamma=5,
                                   grow_policy='lossguide', learning_rate=0.1,
                                   max_bin=255, max_delta_step=0, max_depth=0,
                                   max_leaves=127, min_child_weight=1,
                                   missing=nan, n_estimators=10, n_jobs=-1,
                                   nthread=None, objective='reg:logistic',
                                   random_state=0, reg_alpha=1.9791666666666667,
                                   reg_lambda=0.10416666666666667,
                                   scale_pos_weight=1, seed=None, silent=None,
                                   subsample=0.9, tree_method='hist',
                                   verbose=-10, verbosity=0))],
         verbose=False), '8': Pipeline(memory=None,
         steps=[('sparsenormalizer',
                 <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f12520b5198>),
                ('xgboostclassifier',
                 XGBoostClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=0.8, eta=0.2, gamma=0,
                                   learning_rate=0.1, max_delta_step=0,
                                   max_depth=6, max_leaves=0,
                                   min_child_weight=1, missing=nan,
                                   n_estimators=100, n_jobs=-1, nthread=None,
                                   objective='reg:logistic', random_state=0,
                                   reg_alpha=0, reg_lambda=1.9791666666666667,
                                   scale_pos_weight=1, seed=None, silent=None,
                                   subsample=0.8, tree_method='auto',
                                   verbose=-10, verbosity=0))],
         verbose=False), '1': Pipeline(memory=None,
         steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
                ('xgboostclassifier',
                 XGBoostClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=1, gamma=0,
                                   learning_rate=0.1, max_delta_step=0,
                                   max_depth=3, min_child_weight=1, missing=nan,
                                   n_estimators=100, n_jobs=-1, nthread=None,
                                   objective='binary:logistic', random_state=0,
                                   reg_alpha=0, reg_lambda=1,
                                   scale_pos_weight=1, seed=None, silent=None,
                                   subsample=1, tree_method='auto', verbose=-10,
                                   verbosity=0))],
         verbose=False), '6': Pipeline(memory=None,
         steps=[('sparsenormalizer',
                 <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f12520c0240>),
                ('xgboostclassifier',
                 XGBoostClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=0.5, eta=0.1, gamma=0,
                                   learning_rate=0.1, max_delta_step=0,
                                   max_depth=6, max_leaves=15,
                                   min_child_weight=1, missing=nan,
                                   n_estimators=100, n_jobs=-1, nthread=None,
                                   objective='reg:logistic', random_state=0,
                                   reg_alpha=0, reg_lambda=2.0833333333333335,
                                   scale_pos_weight=1, seed=None, silent=None,
                                   subsample=1, tree_method='auto', verbose=-10,
                                   verbosity=0))],
         verbose=False), '5': Pipeline(memory=None,
         steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
                ('randomforestclassifier',
                 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                        class_weight='balanced',
                                        criterion='entropy', max_depth=None,
                                        max_features='sqrt',
                                        max_leaf_nodes=None, max_samples=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=0.01,
                                        min_samples_split=0.2442105263157895,
                                        min_weight_fraction_leaf=0.0,
                                        n_estimators=10, n_jobs=-1,
                                        oob_score=False, random_state=None,
                                        verbose=0, warm_start=False))],
         verbose=False)}
```

While some algorithms such as XGBoost were used multiple times in the ensemble, there are differences in the parameters such as max_leaf_nodes and max_depth which determine the number splits and depth of the tree. See [here](https://xgboost.readthedocs.io/en/latest/parameter.html "XGBoost parameter tuning") for additional documentation.

## Pipeline comparison

AutoML yielded a result of 0.918 compared to the HyperDrive result of 0.915. The difference in accuracy isn't very significant however.

## Future work
It will be a point of learning to try other models and adjust hyperparameters to compare results. For example we could allow a more lax slack factor with the Bandit Policy to assess the effect on the final result. Although I suspect this will create a marginal improvement in final accuracy.




