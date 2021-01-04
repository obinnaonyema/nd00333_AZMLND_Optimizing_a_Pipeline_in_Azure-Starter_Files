# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data set contains data about customers with which we seek to predict their interests.
The best algorithm was the voting ensemble algorithm with 0.9181 accuracy derived via AutoML run.
## Scikit-learn Pipeline
a. Architecture: VM General Purpose CPU Cluster Compute D-Series V2
b. Data: CSV data source. Data is loaded and cleaned using the train.py script
c. Classification done via Scikit-learn Logistic Regression Model 
d. Hyperparameters: “C” is the regularization parameter, “max-iter” is the maximum number of iterations allowed
**What are the benefits of the parameter sampler you chose?**
RandomParameterSampling allowed us optimize resource usage.
**What are the benefits of the early stopping policy you chose?**
BanditPolicy prevented unnecessary runs where there was no improvement in results after a few consecutive runs as defined.

## AutoML
Multiple models were generated during the AutoML run with Voting Ensemble providing the best result.

## Pipeline comparison
AutoML yielded a result of 0.918 compared to the HyperDrive result of 0.915. The difference in accuracy isn't very significant however.

## Future work
It will be a point of learning to try other models and adjust hyperparameters to compare results.


