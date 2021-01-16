# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree.

In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

## Summary

The data set contains data about customers with which we seek to predict their interests. It contains customer data from a bank's marketing campaigns. ![View data source from Kaggle here](https://www.kaggle.com/yufengsui/portuguese-bank-marketing-data-set?select=bank-full.csv). There are 21 columns and over 32000 rows containing data on employment, marital status, education, housing among others. Some of the columns will need to be one-hot encoded to have numeric values for them.

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



## Pipeline comparison

AutoML yielded a result of 0.918 compared to the HyperDrive result of 0.915. The difference in accuracy isn't very significant however.

## Future work
It will be a point of learning to try other models and adjust hyperparameters to compare results. For example we could allow a more lax slack factor with the Bandit Policy to assess the effect on the final result. Although I suspect this will create a marginal improvement in final accuracy.




