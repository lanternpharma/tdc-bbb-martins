#!/usr/bin/env python
# coding: utf-8

#  Copyright (c) 2023. Lantern Pharma Inc. All rights reserved.

import os

import pickle

# create output folders for the results
method_name = 'ensemble'

results_path = r'../results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

method_path = r'../results/{}/'.format(method_name) 
if not os.path.exists(method_path):
    os.makedirs(method_path)

base_learner_path = r'../results/{}/base_learner_predictions/'.format(method_name) 
if not os.path.exists(base_learner_path):
    os.makedirs(base_learner_path)

test_path = r'../results/{}/test_predictions/'.format(method_name) 
if not os.path.exists(test_path):
    os.makedirs(test_path)

val_path = r'../results/{}/val_predictions/'.format(method_name) 
if not os.path.exists(val_path):
    os.makedirs(val_path)

trained_models_path = r'../results/{}/trained_models/'.format(method_name) 
if not os.path.exists(trained_models_path):
    os.makedirs(trained_models_path)
    
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

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

def forward_selection(df):
    def data_prep(df):   
        # split predictors and targets
        X=df.iloc[:,1:]
        features = list(X.columns)
        y=df.iloc[:,0]
        return X,y, features

    X,y,features=data_prep(df)
    
    model = LogisticRegression()

    sfs = SFS(model, 
          k_features=(2,X.shape[1]), 
          forward=True, 
          floating=False, 
          scoring='neg_mean_absolute_error',
          verbose=2,
          cv=5)

    sfs = sfs.fit(X, y)
    
    features_selected = list(sfs.k_feature_names_)
    print('Best subset (corresponding names):', features_selected)
    print('# Features selected:', len(features_selected))

    features_selected_df = pd.DataFrame(features_selected)

    return features_selected, features_selected_df


def train_logistic(train_df, scale_method):
    X_train,y_train, features_train=data_prep(train_df,scale=scale_method)

    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=global_seed)

    # Perform GridSearchCV to tune best-fit LR model
    param = {
        #'penalty': ['elasticnet'],
        #'penalty': ['l1','l2','elasticnet'],
        #'penalty': ['l2'],
        #'C': np.logspace(-2,0,20),
        'class_weight': ['None', 'balanced'], #['None', 'balanced']
        #'solver': ['saga'],
        'solver': ['saga', 'lbfgs', 'liblinear'],
        #'l1_ratio': np.logspace(-2,0,20),
        #'tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'random_state': [global_seed]
    }
    lr_model = LogisticRegression()
    gs_model = GridSearchCV(estimator=lr_model, param_grid=param, cv=sss, scoring='roc_auc')
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

results_df = pd.DataFrame(columns = ["Model", "Data Set", "AUC", "Accuracy", "f1 Score", "sensitivity", "specificity", "seed"])

for seed in [1, 2, 3, 4, 5]:
    #benchmark = group.get('BBB_Martins')
    #name = benchmark['name']

    #Load validation set base learner predictions to train ensemble model
    #logistic_val_pred = pd.read_csv("../results/logistic/val_predictions/val_predictions_seed{}.tsv".format(seed), sep="\t")
    rf_val_pred = pd.read_csv("../results/random_forest/val_predictions/val_predictions_seed{}.tsv".format(seed), sep="\t")
    dnn_val_pred = pd.read_csv("../results/dnn/val_predictions/val_predictions_seed{}.tsv".format(seed), sep="\t")
    svm_val_pred = pd.read_csv("../results/svm_linear/val_predictions/val_predictions_seed{}.tsv".format(seed), sep="\t")
    base_learner_preds_val = dnn_val_pred.drop('Predicted_class', axis=1).copy()
    base_learner_preds_val.rename(columns={'Predicted_prob': 'dnn_pred'}, inplace=True)
    base_learner_preds_val['rf_pred'] = rf_val_pred.Predicted_prob
    base_learner_preds_val['svm_pred'] = svm_val_pred.Predicted_prob
    #base_learner_preds_val['logistic_pred'] = logistic_val_pred.Predicted_prob
    base_learner_preds_val.set_index(list(base_learner_preds_val.columns[[0]]),inplace=True)
    base_learner_preds_val.to_csv("{}/base_learner_preds_val_seed{}.tsv".format(base_learner_path, seed), sep="\t", index=0)

        
    #Load test set base learner predictions as inputs to the ensemble model
    #logistic_test_pred = pd.read_csv("../results/logistic/test_predictions/test_predictions_seed{}.tsv".format(seed), sep="\t")
    rf_test_pred = pd.read_csv("../results/random_forest/test_predictions/test_predictions_seed{}.tsv".format(seed), sep="\t")
    dnn_test_pred = pd.read_csv("../results/dnn/test_predictions/test_predictions_seed{}.tsv".format(seed), sep="\t")
    svm_test_pred = pd.read_csv("../results/svm_linear/test_predictions/test_predictions_seed{}.tsv".format(seed), sep="\t")
    base_learner_preds_test = dnn_test_pred.drop('Predicted_class', axis=1).copy()
    base_learner_preds_test.rename(columns={'Predicted_prob': 'dnn_pred'}, inplace=True)
    base_learner_preds_test['rf_pred'] = rf_test_pred.Predicted_prob
    base_learner_preds_test['svm_pred'] = svm_test_pred.Predicted_prob
    #base_learner_preds_test['logistic_pred'] = logistic_test_pred.Predicted_prob
    base_learner_preds_test.set_index(list(base_learner_preds_test.columns[[0]]),inplace=True)
    base_learner_preds_test.to_csv("{}/base_learner_preds_test_seed{}.tsv".format(base_learner_path,seed), sep="\t", index=0)

    
    print(base_learner_preds_val.shape)
    print(base_learner_preds_test.shape)

    # Select the base learner models to be used in Ensemble
    #base_learners_selected, baselearner_fs_ranks = forward_selection(base_learner_preds_val)
    #print("# Base Learners Selected: ",len(base_learners_selected))
    #print(baselearner_fs_ranks.head(30))
    #baselearner_fs_ranks.to_csv("{}/baselearner_fs_ranks.tsv".format(results_path), sep="\t", index=0)


    #all_models = base_learner_preds_val.columns[1:]
    #print("All models considered: ",all_models)
    #models_not_selected = set(all_models) - set(base_learners_selected)
    #print("Models eliminated from Ensemble: ",models_not_selected)

    # Use all models rather than drop any through selection
    base_learner_preds_val_selected = base_learner_preds_val.copy()
    base_learner_preds_test_selected = base_learner_preds_test.copy()

    #Two lines commented out below to skip model selection    
    #base_learner_preds_val_selected = base_learner_preds_val.copy().drop(models_not_selected, axis=1)
    #base_learner_preds_test_selected = base_learner_preds_test.copy().drop(models_not_selected, axis=1)
    print("Models used in ensemble: ", base_learner_preds_val_selected.columns[1:])

    #Train the ensemble model with validation set base learner predicted probabilities
    #ensemble_model = train_logistic(base_learner_preds_val_selected, "None")
    ensemble_model = train_logistic(augment_data(base_learner_preds_val_selected), "None")

    #Train the ensemble model with validation set base learner predicted probabilities
    #ensemble_model = train_logistic(augment_data(base_learner_preds_val), "None")

    #Make predictions on the test set
    predictions = {}
    #name = benchmark['name']
    name = 'BBB_Martins'
    predicted_prob, predicted_class = logistic_predict(ensemble_model, base_learner_preds_test_selected, "None")
    predictions[name] = predicted_prob

    with open("../results/{}/trained_models/model_seed_{}.pkl".format(method_name, seed), "wb") as pkl:
        pickle.dump(ensemble_model, pkl)

    predictions_list.append(predictions)
    
    # write output files
    # write the above dataframe as .tsv file to use for feature selection
    test_predictions_df = pd.DataFrame({'Drug_ID':base_learner_preds_test.index.values, 'Actual_value':base_learner_preds_test.Actual_value, 'Predicted_prob':predicted_prob, 'Predicted_class':predicted_class})
    test_predictions_df.to_csv("../results/{}/test_predictions/test_predictions_seed{}.tsv".format(method_name, seed), sep="\t", index=0)

    test_auc = roc_auc_score(base_learner_preds_test['Actual_value'], predicted_prob, average='weighted')
    print("\n Test AUC seed {}: ".format(seed),test_auc, "\n")

    def get_metrics(actual,predicted_prob, predicted_class):
        AUC = roc_auc_score(actual, predicted_prob)
        Accuracy = accuracy_score(actual, predicted_class)
        f1 = f1_score(actual, predicted_class)
        sensitivity = recall_score(actual, predicted_class, pos_label=1)
        specificity = recall_score(actual, predicted_class, pos_label=0)
        return AUC, Accuracy, f1, sensitivity, specificity

    test_AUC, test_Accuracy, test_f1, test_sensitivity, test_specificity = get_metrics(test_predictions_df.Actual_value,                test_predictions_df.Predicted_prob, test_predictions_df.Predicted_class)
    print("test AUC: ", test_AUC)
    print("test Accuracy: ",test_Accuracy)
    print("testf1 score: ", test_f1)
    print("sensitivity score: ", test_sensitivity)
    print("specificity score: ", test_specificity)
    
    results_df.loc[len(results_df.index)] = ["Ensemble", "test", test_AUC, test_Accuracy, test_f1, test_sensitivity, test_specificity, seed]

    val_predicted_prob, val_predicted_class = logistic_predict(ensemble_model, base_learner_preds_val_selected, "None")

    # write the above dataframe as .tsv file to use for feature selection
    val_predictions_df = pd.DataFrame({'Drug_ID':base_learner_preds_val.index.values, 'Actual_value':base_learner_preds_val.Actual_value, 'Predicted_prob':val_predicted_prob, 'Predicted_class':val_predicted_class})
    val_predictions_df.to_csv("../results/{}/val_predictions/val_predictions_seed{}.tsv".format(method_name, seed), sep="\t", index=0)


    
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

results_df.to_csv("../results/{}/model_performance.tsv".format(method_name), sep="\t", index=0)



