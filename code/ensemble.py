#!/usr/bin/env python
# coding: utf-8

#  Copyright (c) 2023. Lantern Pharma Inc. All rights reserved.

import os

# create output folders for the results
method_name = 'ensemble'

results_path = r'../results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

method_path = r'../results/{}/'.format(method_name) 
if not os.path.exists(method_path):
    os.makedirs(method_path)

test_path = r'../results/{}/test_predictions/'.format(method_name) 
if not os.path.exists(test_path):
    os.makedirs(test_path)

    
import numpy as np
import pandas as pd
from tdc.single_pred import ADME
import random
from tdc.benchmark_group import admet_group

from imblearn.over_sampling import SVMSMOTE

from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit, RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import datetime
now = datetime.datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H:%M:%S")
print ("Current date and time : ", time_stamp)

global_seed = 8516

def data_prep(df,scale):
    # split predictors and targets
    X=df.iloc[:,1:]
    features = list(X.columns)
    y=df.iloc[:,0]
    # scale X
    if (scale=="MaxAbsScaler"):
        Xs=MaxAbsScaler().fit_transform(X)
    elif (scale=="StandardScaler"):
        Xs=StandardScaler().fit_transform(X)
    elif (scale=="MinMaxScaler"):
        Xs=MinMaxScaler().fit_transform(X)
    elif (scale=="RowSums"):
        Xs=(X.T / X.T.sum()).T
    else:
        Xs=X
    return Xs,y, features

def augment_data(df):
    print("Before OverSampling, counts of label '0': {}".format(sum(df.iloc[:,0] == 0)))
    print("Before OverSampling, counts of label '1': {} \n".format(sum(df.iloc[:,0] == 1)))
    sm = SVMSMOTE(random_state = 8516)
    X_res, y_res = sm.fit_resample(df.iloc[:,1:], df.iloc[:,0].ravel())
    y_res = pd.DataFrame(y_res, columns=['target'])
    resampled_train_data = pd.DataFrame(pd.concat([y_res, X_res], axis=1))
    print("After OverSampling, counts of label '0': {}".format(sum(resampled_train_data.iloc[:,0] == 0)))
    print("After OverSampling, counts of label '1': {} \n".format(sum(resampled_train_data.iloc[:,0] == 1)))
    return resampled_train_data

def train_logistic(train_df, scale_method):
    X_train,y_train, features_train=data_prep(train_df,scale=scale_method)

    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=global_seed)

    # Perform GridSearchCV to tune best-fit LR model
    param = {
        #'penalty': ['elasticnet'],
        'penalty': ['l1','l2','elasticnet'],
        'C': np.logspace(-2,0,20),
        'class_weight': ['None', 'balanced'], #['None', 'balanced']
        'solver': ['saga'],
        #'solver': ['saga', 'lbfgs', 'liblinear'],
        'l1_ratio': np.logspace(-2,0,20),
        #'tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'random_state': [global_seed]
    }
    lr_model = LogisticRegression()
    gs_model = GridSearchCV(estimator=lr_model, param_grid=param, cv=sss)
    gs_model.fit(X_train, y_train)
    print("Best Parameters from search: ",gs_model.best_params_)
    # Train a LR model with best parameters
    lr_model = LogisticRegression(**gs_model.best_params_)
    lr_model.fit(X_train, y_train)
    print("Ensemble Model Intercept: ",lr_model.intercept_)
    print("Ensemble Feature Weights: ",lr_model.coef_)
    print(features_train)
    
    return lr_model

# Function to make predictions 

def logistic_predict(trained_model, df, scale_method):
    #split features and targets for modeling
    X_test,y_test, features_test=data_prep(df,scale=scale_method)
    model_prediction_prob = trained_model.predict_proba(X_test)
    model_prediction_prob = model_prediction_prob[:,1]
    model_prediction_class = [1 if pred > 0.5 else 0 for pred in model_prediction_prob]
    return model_prediction_prob, model_prediction_class

### Get the TDC train and test split Drug_IDs to split similarly
group = admet_group(path = 'data/')
predictions_list = []

### use the code provided in the TDC website
#### https://tdcommons.ai/benchmark/overview/

for seed in [1, 2, 3, 4, 5]:
    #benchmark = group.get('BBB_Martins')
    #name = benchmark['name']

    #Load validation set base learner predictions to train ensemble model
    logistic_val_pred = pd.read_csv("../results/logistic_fe_aug/val_predictions/val_predictions_seed{}.tsv".format(seed), sep="\t")
    #rf_val_pred = pd.read_csv("../results/rf_fe_aug/val_predictions/val_predictions_seed{}.tsv".format(seed), sep="\t")
    dnn_val_pred = pd.read_csv("../results/dnn_fe_aug/val_predictions/val_predictions_seed{}.tsv".format(seed), sep="\t")
    base_learner_preds_val = logistic_val_pred.drop('Predicted_class', axis=1).copy()
    base_learner_preds_val.rename(columns={'Predicted_prob': 'logistic_pred'}, inplace=True)
    #base_learner_preds_val['rf_pred'] = rf_val_pred.Predicted_prob
    base_learner_preds_val['dnn_pred'] = dnn_val_pred.Predicted_prob
    base_learner_preds_val.set_index(list(base_learner_preds_val.columns[[0]]),inplace=True)
        
    #Load test set base learner predictions as inputs to the ensemble model
    logistic_test_pred = pd.read_csv("../results/logistic_fe_aug/test_predictions/test_predictions_seed{}.tsv".format(seed), sep="\t")
    #rf_test_pred = pd.read_csv("../results/rf_fe_aug/test_predictions/test_predictions_seed{}.tsv".format(seed), sep="\t")
    dnn_test_pred = pd.read_csv("../results/dnn_fe_aug/test_predictions/test_predictions_seed{}.tsv".format(seed), sep="\t")
    base_learner_preds_test = logistic_test_pred.drop('Predicted_class', axis=1).copy()
    base_learner_preds_test.rename(columns={'Predicted_prob': 'logistic_pred'}, inplace=True)
    #base_learner_preds_test['rf_pred'] = rf_test_pred.Predicted_prob
    base_learner_preds_test['dnn_pred'] = dnn_test_pred.Predicted_prob
    base_learner_preds_test.set_index(list(base_learner_preds_test.columns[[0]]),inplace=True)
    
    #Train the ensemble model with validation set base learner predicted probabilities
    ensemble_model = train_logistic(augment_data(base_learner_preds_val), "None")

    #Make predictions on the test set
    predictions = {}
    #name = benchmark['name']
    name = 'BBB_Martins'
    predicted_prob, predicted_class = logistic_predict(ensemble_model, base_learner_preds_test, "None")
    predictions[name] = predicted_prob

    predictions_list.append(predictions)
    
    # write output files
    # write the above dataframe as .tsv file to use for feature selection
    test_predictions_df = pd.DataFrame({'Drug_ID':base_learner_preds_test.index.values, 'Actual_value':base_learner_preds_test.Actual_value, 'Predicted_prob':predicted_prob, 'Predicted_class':predicted_class})
    test_predictions_df.to_csv("../results/{}/test_predictions/test_predictions_seed{}.tsv".format(method_name, seed), sep="\t", index=0)

    test_auc = roc_auc_score(base_learner_preds_test['Actual_value'], predicted_prob, average='weighted')
    print("\n Test AUC seed {}: ".format(seed),test_auc, "\n")
    
print(len(predictions_list))
results = group.evaluate_many(predictions_list)
print(results)

f = open("../results/{}/tdc_performance_results.txt".format(method_name), "w")
f.write("Model: {}".format(method_name))
f.write("Results from the TDC group.evaluate_many(predictions_list) function: \n")
f.write(str(results))
f.write("\n Average AUC, standard deviation \n")
f.close()

f = open("../results/tdc_results_summary.txt", "a")
f.write("Model: {} \n".format(method_name))
f.write("Results from the TDC group.evaluate_many(predictions_list) function: \n")
f.write(str(results))
f.write("\n Average AUC, standard deviation \n")
f.close()
