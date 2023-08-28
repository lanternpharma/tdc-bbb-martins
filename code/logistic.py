#Relative paths to data inputs used by this script
data_path = '/code/data/tdc_bbb_martins_ml_data/'
corr_feature_path = '/code/data/correlated_features/'

# Required package imports

import numpy as np
import pandas as pd
from tdc.single_pred import ADME
import random
from tdc.benchmark_group import admet_group


from imblearn.over_sampling import SVMSMOTE
from sklearn.feature_selection import RFECV

from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit, RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss, confusion_matrix

from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import pickle
import os 

import datetime
now = datetime.datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H:%M:%S")
print ("Current date and time : ", time_stamp)

# Load TensorFlow for neural networks and set seeds for reproducibility
# Set a seed value
global_seed= 8516
random.seed(global_seed)
np.random.seed(global_seed)
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(global_seed)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(global_seed)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(global_seed)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(global_seed)

#5 Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

#6 The determinism ection below ensures that the nn results will be repeatable given the same inputs
# This may slow down performance. If you want to do many runs to look at average performance this should be removed
tf.keras.utils.set_random_seed(global_seed)
tf.config.experimental.enable_op_determinism()
print(tf.__version__)
from keras.callbacks import History, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# create output folders for the results
method_name = 'logistic'

results_path = r'../results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

method_path = r'../results/{}/'.format(method_name) 
if not os.path.exists(method_path):
    os.makedirs(method_path)

fs_path = r'../results/{}/feature_selection/'.format(method_name) 
if not os.path.exists(fs_path):
    os.makedirs(fs_path)

test_path = r'../results/{}/test_predictions/'.format(method_name) 
if not os.path.exists(test_path):
    os.makedirs(test_path)

valid_path = r'../results/{}/val_predictions/'.format(method_name) 
if not os.path.exists(valid_path):
    os.makedirs(valid_path)

trainval_path = r'../results/{}/trainval_oof_predictions/'.format(method_name) 
if not os.path.exists(trainval_path):
    os.makedirs(trainval_path)

trained_models_path = r'../results/{}/trained_models/'.format(method_name) 
if not os.path.exists(trained_models_path):
    os.makedirs(trained_models_path)

# Load data created by generate_features.py

### Get trainval features that were generated in previous script
full_train_data = pd.read_csv('{}trainval.tsv'.format(data_path), sep="\t")
#Asset included index as first column, remove it before further processing
full_train_data = full_train_data.iloc[: , 1:]
print("Full Train/Val data shape: ",full_train_data.shape)

### Get trainval features that were generated in previous script
full_test_data = pd.read_csv('{}test.tsv'.format(data_path), sep="\t")
#Asset included index as first column, remove it before further processing
full_test_data = full_test_data.iloc[: , 1:]
print("Full Test data shape: ",full_train_data.shape)

# 3) Prepare data (add sig fingerprint features, scale data, format df for ml functions)
# Create function to move Drug_ID to index, and re-format the target from categorical to numerical

def format_df_to_ml(df):
    # move id column to index
    df.set_index(list(df.columns[[0]]),inplace=True)
    # convert the sting to category
    df.iloc[:, 0] = pd.Categorical(df.iloc[:, 0])
    # creating the mapping dictionary
    replace_target_pred = dict( enumerate(df.iloc[:, 0].cat.categories ) )
    replace_target = dict( enumerate(df.iloc[:, 0].cat.categories ) )
    # reverse the dictionary (key value reverse)
    replace_target = {v: k for k, v in replace_target.items()}
    print ("Target value mapping: {}".format(replace_target))
    # replace the target value with numbers to avoid the error further in fitting
    df.iloc[:, 0] = df.iloc[:, 0].map(replace_target)
    # convert the category type column to integer
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0])
    return df, replace_target

# Create a function to relabel the target from numerical to original mapping
def relabel_target(df, dictionary):
    dictionary_reverse = {v: k for k, v in dictionary.items()}
    df=df.replace({"target": dictionary_reverse})

#Add function for SMOTE-SVM data augmentation

def augment_data(df):
    print("Before OverSampling, counts of label '0': {}".format(sum(df.iloc[:,0] == 0)))
    print("Before OverSampling, counts of label '1': {} \n".format(sum(df.iloc[:,0] == 1)))
    sm = SVMSMOTE(random_state = 8516)
    X_res, y_res = sm.fit_resample(df.iloc[:,1:], df.iloc[:,0].ravel())
    y_res = pd.DataFrame(y_res, columns=['target'])
    resampled_train_data = pd.DataFrame(pd.concat([y_res, X_res], axis=1))
    print("After OverSampling, counts of label '0': {}".format(sum(resampled_train_data.iloc[:,0] == 0)))
    print("After OverSampling, counts of label '1': {} \n".format(sum(resampled_train_data.iloc[:,0] == 1)))

    num_synthetic = resampled_train_data.shape[0] - df.shape[0]
    print("num_synthetic ",num_synthetic)
    
    aug_index = list(df.index.copy())
    #print("aug_index before adding synthetic: ",aug_index)

    for i in range(num_synthetic):
        aug_index.append("synthetic_{}".format(i))
    
    #print("aug_index after adding synthetic: ",aug_index)

    resampled_train_data.index = list(aug_index)
    return resampled_train_data

# function to split target and features for ML inputs

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

# Create Scaler functions with saved pickle file
def fit_scaler(df,scale):
    from pickle import dump
    # split predictors and targets
    X=df.iloc[:,1:]
    features = list(X.columns)
    y=df.iloc[:,0]
    # scale X
    if (scale=="MaxAbsScaler"):
        scaler = MaxAbsScaler()
        scaler.fit(X[X.columns])
    elif (scale=="StandardScaler"):
        scaler = StandardScaler()
        scaler.fit(X[X.columns])
    elif (scale=="MinMaxScaler"):
        scaler = MinMaxScaler()
        scaler.fit(X[X.columns])
    dump(scaler, open('../results/{}/scaler.pkl'.format(method_name), 'wb'))
    

def scaler_transform(df, scaler):
    from pickle import load
    scaler = load(open(scaler, 'rb'))
    # split predictors and targets
    X=df.iloc[:,1:]
    features = list(X.columns)
    y=df.iloc[:,0]
    Xs = scaler.transform(X[X.columns])
    Xs_df = pd.DataFrame(Xs, columns=features)
    Xs_df.index = list(y.index)
    return pd.concat([y, Xs_df], axis=1)



# Create function to generate lists of significant fingerprints

def get_fingerprint_lists(df):
    # Create new features for count of significant negatively & positively associated fingerprints
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    rdk_features = list(df.filter(like='RDK').columns)
    maccs_features = list(df.filter(like='MACCS').columns)
    morgan_features = list(df.filter(like='Morgan').columns)
    fingerprint_features = rdk_features + maccs_features + morgan_features

    df.loc[df["target"] == "Non_permeable", "target"] = 0
    df.loc[df["target"] == "Permeable", "target"] = 1
    df.target = df.target.astype('int64')
    
    X_train_fingerprints = df[fingerprint_features]
    y_train = df.target

    chi2_fingerprints = SelectKBest(score_func=chi2, k='all')
    chi2_fingerprints.fit(X_train_fingerprints, y_train)

    chi2_fingerprints_df = pd.DataFrame(list(zip(fingerprint_features, chi2_fingerprints.scores_,           chi2_fingerprints.pvalues_)), columns=['Feature','Chi2','p_value'])

    fp_percent_permeable = []
    for i in fingerprint_features:
        if (df[i].max()==0):
            fp_percent_permeable.append(np.nan)
        else:
            fp_percent_permeable.append(df.groupby(i)['target'].mean()[1])

    num_with_fingerpint = []
    for i in fingerprint_features:
        num_with_fingerpint.append(df[str(i)].sum())
    print(len(num_with_fingerpint))

    chi2_fingerprints_df['percent_permeable'] = fp_percent_permeable
    chi2_fingerprints_df['count_with_fingerprint'] = num_with_fingerpint

    neg_sig_fingerprints = list(chi2_fingerprints_df[(chi2_fingerprints_df.percent_permeable < 0.40) & (chi2_fingerprints_df.p_value < 0.05)].Feature)

    pos_sig_fingerprints = list(chi2_fingerprints_df[(chi2_fingerprints_df.percent_permeable > 0.80) & (chi2_fingerprints_df.p_value < 0.05)].Feature)
    return neg_sig_fingerprints, pos_sig_fingerprints

# Create funtion to take sig fingerprint lists and add features for count of sig fingerprints

def get_num_fingerprints(df, neg_sig_fingerprints, pos_sig_fingerprints):
    df['neg_sig_fingerprints'] = df[neg_sig_fingerprints].sum(axis=1)
    df['pos_sig_fingerprints'] = df[pos_sig_fingerprints].sum(axis=1)
    return df

# Format full_train_data & full_test_data to ML ready (ID moved to index, target mapped to binary)
full_train_data_ml, target_dictionary = format_df_to_ml(full_train_data)
full_test_data_ml, target_dictionary = format_df_to_ml(full_test_data)

# Calculate neg_sig_fingerprints, pos_sig_fingerprints on full_train_data then add to full_test_data
neg_sig_fingerprints, pos_sig_fingerprints = get_fingerprint_lists(full_train_data_ml)

neg_sig_fingerprints_df = pd.DataFrame(neg_sig_fingerprints, columns=['neg_sig_fingerprints'])
neg_sig_fingerprints_df.to_csv("../results/{}/neg_sig_fingerprints.tsv".format(method_name), sep="\t", index=0)

pos_sig_fingerprints_df = pd.DataFrame(pos_sig_fingerprints, columns=['pos_sig_fingerprints'])
pos_sig_fingerprints_df.to_csv("../results/{}/pos_sig_fingerprints.tsv".format(method_name), sep="\t", index=0)

full_train_data_ml = get_num_fingerprints(full_train_data_ml, neg_sig_fingerprints, pos_sig_fingerprints)
full_test_data_ml = get_num_fingerprints(full_test_data_ml, neg_sig_fingerprints, pos_sig_fingerprints)

print("full_train_data_ml shape: ",full_train_data_ml.shape)
print("full_test_data_ml shape: ",full_test_data_ml.shape)

# 4) Impute missing values
# Impute full_train_data NAN with column means, the fit imputer on full_train_data to impute full_test_data
#Replace inf values in training set with NAN prior to imputing
full_train_data_ml.replace([np.inf, -np.inf], np.nan, inplace=True)
full_train_data_ml[full_train_data_ml > 1e+5] = np.nan
full_train_data_ml[full_train_data_ml < -1e+5] = np.nan
print("full_train_data_ml NAN: ", full_train_data_ml.isnull().any(axis=1).sum())


#Rplace NAN values with column mean in training prior to fitting imputer for validation and test sets
for i in full_train_data_ml.columns[full_train_data_ml.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
    full_train_data_ml[i].fillna(full_train_data_ml[i].mean(),inplace=True)
    
print("full_train_data_ml NAN after imputing: ", full_train_data_ml.isnull().any(axis=1).sum())


#Replace any infinite values in test set with NAN before imputing
full_test_data_ml.replace([np.inf, -np.inf], np.nan, inplace=True)
full_test_data_ml[full_test_data_ml > 1e+5] = np.nan
full_test_data_ml[full_test_data_ml < -1e+5] = np.nan
print("full_test_data NAN: ", full_test_data_ml.isnull().any(axis=1).sum())
    
#Replace NAN in test with imputer fit to training set
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
imputer = imputer.fit(full_train_data_ml.iloc[: , 1:])
full_test_data_imputed = pd.DataFrame(imputer.transform(full_test_data_ml.iloc[: , 1:]), columns=full_test_data_ml.columns[1:], index=full_test_data.index)
full_test_data_imputed = pd.concat([full_test_data.target, full_test_data_imputed], axis=1)
print("full_test_data_imputed NAN: ", full_test_data_imputed.isnull().any(axis=1).sum())
print("full_test_data_imputed shape: ", full_test_data_imputed.shape)
    
#Replace full_test_data_ml with imputed version
full_test_data_ml = full_test_data_imputed

# 5) Scale data

# Fit scaler on full_train_data (trainval combined) then transform test set too
fit_scaler(full_train_data_ml, "StandardScaler")
full_train_data_ml_scaled = scaler_transform(full_train_data_ml, '../results/{}/scaler.pkl'.format(method_name))
full_test_data_ml_scaled = scaler_transform(full_test_data_ml, '../results/{}/scaler.pkl'.format(method_name)) 

full_train_data_ml_scaled.head()

# Drop highly correlated features (tanimoto or R > 0.95) using lists created by correlated_features.py

with open("{}correlated_fingerprints_to_drop.pkl".format(corr_feature_path), "rb") as fp:
    correlated_fingerprints_to_drop = pickle.load(fp)

with open("{}correlated_numeric_to_drop.pkl".format(corr_feature_path), "rb") as num_corr:
    correlated_numeric_to_drop = pickle.load(num_corr)

print("full_train_data_ml_scaled columns: ",len(full_train_data_ml_scaled.columns))

full_train_data_ml_scaled.drop(correlated_fingerprints_to_drop , axis=1, inplace=True)
print("Full Train/Val data shape after dropping correlated fingerprints: ",full_train_data_ml_scaled.shape)

full_test_data_ml_scaled.drop(correlated_fingerprints_to_drop , axis=1, inplace=True)
print("Full Test data shape after dropping correlated fingerprints: ",full_test_data_ml_scaled.shape)

full_train_data_ml_scaled.drop(correlated_numeric_to_drop, axis=1, inplace=True)
full_test_data_ml_scaled.drop(correlated_numeric_to_drop, axis=1, inplace=True)
print("Full Train/Val data shape after dropping correlated numeric features: ",full_train_data_ml_scaled.shape)
print("Full Test data shape after dropping correlated numeric features: ",full_test_data_ml_scaled.shape)

# fit kpca on trainval, transform trainval and test

from sklearn.decomposition import KernelPCA

def kpca_poly_fit(df):
    X=df.iloc[:,1:]
    features = list(X.columns)
    y=pd.DataFrame(df.iloc[:,0])
    kpca = KernelPCA(kernel='poly', random_state=global_seed, remove_zero_eig=True, degree=2, gamma=0.001) # , n_components=500
    #kpca = KernelPCA(kernel='linear', random_state=global_seed, remove_zero_eig=True) 
    kpca.fit(X)
    from pickle import dump
    dump(kpca, open('../results/{}/kpca_poly.pkl'.format(method_name), 'wb'))
    return kpca
    
def kpca_poly_transform(kpca, df):
    X=df.iloc[:,1:]
    features = list(X.columns)
    y=pd.DataFrame(df.iloc[:,0])
    print("y shape: ", y.shape)
    X_pca = kpca.transform(X)
    col = []
    # read about what features used if found.
    for i in range(X_pca.shape[1]):
        col.append('kpca_' + str(i))
    df_pca = pd.DataFrame(X_pca, columns=col)
    df_pca.index = y.index
    print("df_pca shape: ",df_pca.shape)
    df_transformed = pd.concat([y, df_pca], axis=1)
    return df_transformed

# Perform KPCA on full_train_data (trainval combined), then transform full_test_data
kpca_poly = kpca_poly_fit(full_train_data_ml_scaled)
#kpca_poly = kpca_poly_fit(full_train_data_ml_augmented_pre_split)

full_train_data_ml_kpca = kpca_poly_transform(kpca_poly, full_train_data_ml_scaled)
full_test_data_ml_kpca = kpca_poly_transform(kpca_poly, full_test_data_ml_scaled)

# 8) feature selection on trainval kpca

def lasso_selection(df, scale_method):
    X,y,features=data_prep(df,scale=scale_method)
    
    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.125, random_state=global_seed)
    
    lr_model = LogisticRegression(penalty='l1')
    
    param_broad = {
    'C': np.arange(0.00001, 1, 0.05), 
    'solver': ['liblinear'],
    'class_weight': ['balanced'],
    'max_iter': [200],
    'random_state': [global_seed]
    }
    
    gs_broad = GridSearchCV(estimator=lr_model, param_grid=param_broad, cv=sss, n_jobs=-1, pre_dispatch=20, scoring='neg_log_loss', verbose=1) #scoring='roc_auc'
    gs_broad.fit(X, y)
    print("Best Parameters from broad search: ",gs_broad.best_params_)
    broad_c = gs_broad.best_params_['C']

    param_refined = {
    'C': np.arange(broad_c-0.05, broad_c+0.05, 0.01),
    'solver': ['liblinear'], 
    'class_weight': ['balanced'],
    'max_iter': [200],
    'random_state': [global_seed]
    }
    
    gs_refined = GridSearchCV(estimator=lr_model, param_grid=param_refined, cv=sss, n_jobs=-1, pre_dispatch=20, scoring='neg_log_loss', verbose=1) #scoring='roc_auc'
    gs_refined.fit(X, y)
    print("Best Parameters from refined search: ",gs_refined.best_params_)
    refined_c = gs_refined.best_params_['C']
    
    param_final = {
    'C': np.arange(refined_c-0.01, refined_c+0.01, 0.001), 
    'solver': ['liblinear'],
    'class_weight': ['balanced'],
    'max_iter': [200],
    'random_state': [global_seed]
    }
    
    gs_final = GridSearchCV(estimator=lr_model, param_grid=param_final, cv=sss, n_jobs=-1, pre_dispatch=20, scoring='neg_log_loss', verbose=1) #scoring='roc_auc'
    gs_final.fit(X, y)
    print("Best Parameters from final search: ",gs_final.best_params_)
    
    # Train a LR model with best parameters
    model = LogisticRegression(**gs_final.best_params_, penalty='l1')
    model.fit(X, y)
    coef = model.coef_[0]
    rankings = pd.DataFrame(list(zip(features, coef)), columns=['feature','coef'])
    rankings.sort_values(by='coef', ascending=False, key=abs, inplace=True)
    rankings = rankings[rankings.coef!=0]
    rankings['Rank'] = range(1,len(rankings.feature)+1,1)
    features_selected = list(rankings.feature)
    return features_selected, rankings

# feature selection on trainval kpca
lasso_features_selected, lasso_fs_ranks = lasso_selection(full_train_data_ml_kpca, "None")
print("Lasso features selected: ",len(lasso_features_selected))
print(lasso_fs_ranks.head())
lasso_fs_ranks.to_csv("../results/{}/feature_selection/lasso_fs_ranks.tsv".format(method_name), sep="\t", index=0)


# 9) model hyperparameter search on trainval
    
def logistic_hyper_search(train_df, scale_method):
    X_train,y_train, features_train=data_prep(train_df,scale=scale_method)
    print("hyper search X_train shape: ", X_train.shape)

    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.125, random_state=global_seed)
    
    # Perform GridSearchCV to tune best-fit LR model
    param = {
        'penalty': ['l2'],
        'C': (list(np.logspace(-2, 1, 30))),
        #'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
        'class_weight': [None, 'balanced'],
        'solver': ['liblinear', 'lbfgs'],
        'random_state': [global_seed]
    }
    
    lr_model = LogisticRegression()
    gs_model = GridSearchCV(estimator=lr_model, param_grid=param, cv=sss, scoring='neg_log_loss', n_jobs=-1, pre_dispatch=20, error_score=0.001, verbose=1)
    gs_model.fit(X_train, y_train)
    print("Best Score: ", gs_model.best_score_)
    print("Best Parameters from search: ",gs_model.best_params_)
    
    # Train a LR model with best parameters
    final_model = LogisticRegression(**gs_model.best_params_)
    
    #grid_results = pd.DataFrame()
    #grid_results['rank'] = gs_model.cv_results_['rank_test_score']
    #grid_results['mean_score'] = gs_model.cv_results_['mean_test_score']
    #grid_results['std_score'] = gs_model.cv_results_['std_test_score']
    #grid_results['params'] = gs_model.cv_results_['params']
    #grid_results = pd.concat([grid_results, grid_results.params.apply(pd.Series)], axis=1)
    #grid_results.drop('params', axis=1, inplace=True)
    #grid_results.sort_values(by='rank', ascending=True, inplace=True)
    #print(grid_results.head(10))

    return final_model
        
# Function to make predictions 

def logistic_fit(model, df, scale_method):
    X_train,y_train, features_train=data_prep(df,scale=scale_method)
    print("fit X_train shape: ", X_train.shape)

    model.fit(X_train, y_train)
    return model
        
def logistic_predict(trained_model, df, scale_method):
    #split features and targets for modeling
    X_test,y_test, features_test=data_prep(df,scale=scale_method)
    print("predict X_test shape: ", X_test.shape)

    model_prediction_prob = trained_model.predict_proba(X_test)
    model_prediction_prob = model_prediction_prob[:,1]
    model_prediction_class = [1 if pred > 0.5 else 0 for pred in model_prediction_prob]
    return model_prediction_prob, model_prediction_class

#Run hyperparameter search on trainval set
best_model = logistic_hyper_search(full_train_data_ml_kpca[['target'] + lasso_features_selected], "None")

# 10) fit model (with best trainval hyper params) with train only for each seed

#fit model (with best trainval hyper params) with train only for each seed

### Get the TDC train and test split Drug_IDs to split similarly
group = admet_group(path = 'data/')
predictions_list = []

results_df = pd.DataFrame(columns = ["Model", "Data Set", "AUC", "Accuracy", "f1 Score", "sensitivity", "specificity", "seed"])

for seed in [1, 2, 3, 4, 5]:
    benchmark = group.get('BBB_Martins')
    name = benchmark['name']
    
    # get the Drug ID's for train/valid/test splits
    train_split, test_split = benchmark['train_val'], benchmark['test']
    train_split, valid_split = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
    train_data_Ids = train_split['Drug_ID']
    valid_data_Ids = valid_split['Drug_ID']
    test_data_Ids = test_split['Drug_ID']
    
    train_df = full_train_data_ml_kpca[full_train_data_ml_kpca.index.isin(train_data_Ids)]
    valid_df = full_train_data_ml_kpca[full_train_data_ml_kpca.index.isin(valid_data_Ids)]
    test_df = full_test_data_ml_kpca[full_test_data_ml_kpca.index.isin(test_data_Ids)]


    best_model_fit = logistic_fit(best_model, train_df[['target'] + lasso_features_selected], "None")
    
    #Make predictions on the test set
    predictions = {}
    name = benchmark['name']
    predicted_prob, predicted_class = logistic_predict(best_model_fit, test_df[['target'] + lasso_features_selected], "None")
    predictions[name] = predicted_prob

    with open("../results/{}/trained_models/model_seed_{}.pkl".format(method_name, seed), "wb") as pkl:
        pickle.dump(best_model_fit, pkl)

    predictions_list.append(predictions)
    
    test_auc = roc_auc_score(full_test_data_ml['target'], predicted_prob, average='weighted')
    print("\n Test AUC seed {}: ".format(seed),test_auc, "\n")
    
    # write output files
    # write the above dataframe as .tsv file to use for feature selection
    test_predictions_df = pd.DataFrame({'Drug_ID':test_df.index.values, 'Actual_value':test_df.target, 'Predicted_prob':predicted_prob, 'Predicted_class':predicted_class})
    test_predictions_df.to_csv("../results/{}/test_predictions/test_predictions_seed{}.tsv".format(method_name, seed), sep="\t", index=0)
    
    def get_metrics(actual,predicted_prob, predicted_class):
        AUC = roc_auc_score(actual, predicted_prob)
        Accuracy = accuracy_score(actual, predicted_class)
        f1 = f1_score(actual, predicted_class)
        sensitivity = recall_score(actual, predicted_class, pos_label=1)
        specificity = recall_score(actual, predicted_class, pos_label=0)
        return AUC, Accuracy, f1, sensitivity, specificity

    test_AUC, test_Accuracy, test_f1, test_sensitivity, test_specificity = get_metrics(test_predictions_df.Actual_value, test_predictions_df.Predicted_prob, test_predictions_df.Predicted_class)
    print("test AUC: ", test_AUC)
    print("test Accuracy: ",test_Accuracy)
    print("testf1 score: ", test_f1)
    print("sensitivity score: ", test_sensitivity)
    print("specificity score: ", test_specificity)


    results_df.loc[len(results_df.index)] = ["Logistic", "test", test_AUC, test_Accuracy, test_f1, test_sensitivity, test_specificity, seed]

    #Make predictions on the validation set
    val_predicted_prob, val_predicted_class = logistic_predict(best_model_fit, valid_df[['target'] + lasso_features_selected], "None")
    # write the above dataframe as .tsv file 
    val_predictions_df = pd.DataFrame({'Drug_ID':valid_df.index.values, 'Actual_value':valid_df.target, 'Predicted_prob':val_predicted_prob, 'Predicted_class':val_predicted_class})
    val_predictions_df.to_csv("../results/{}/val_predictions/val_predictions_seed{}.tsv".format(method_name,seed), sep="\t", index=0)

    val_AUC, val_Accuracy, val_f1, val_sensitivity, val_specificity = get_metrics(val_predictions_df.Actual_value, val_predictions_df.Predicted_prob, val_predictions_df.Predicted_class)
    print("val AUC: ", val_AUC)
    print("val Accuracy: ",val_Accuracy)
    print("val f1 score: ", val_f1)
    print("sensitivity score: ", val_sensitivity)
    print("specificity score: ", val_specificity)

    results_df.loc[len(results_df.index)] = ["Logistic", "validation", val_AUC, val_Accuracy, val_f1, val_sensitivity, val_specificity, seed]

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
