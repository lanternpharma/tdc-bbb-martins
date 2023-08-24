# 1) Required package imports

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

import optuna

import pickle
import os 

import datetime
now = datetime.datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H:%M:%S")
print ("Current date and time : ", time_stamp)

# Load TensorFlow for neural networks and set seeds for reproducibility
# Set a seed value
global_seed= 8516

import tensorflow as tf
    
def reset_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
    tf.keras.utils.set_random_seed(seed_value)
    
reset_seed(global_seed)

#5 Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

#6 The determinism ection below ensures that the nn results will be repeatable given the same inputs
# This may slow down performance. If you want to do many runs to look at average performance this should be removed
#tf.keras.utils.set_random_seed(global_seed)
tf.config.experimental.enable_op_determinism()
print(tf.__version__)
from keras.callbacks import History, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers, initializers
import keras_tuner as kt

keras.initializers.RandomUniform(seed=global_seed)

# create output folders for the results
method_name = 'dnn'

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

# 2) Load data created by generate_features.py

### Get trainval features that were generated in previous script
full_train_data = pd.read_csv('/data/tdc_bbb_martins_ml_data/trainval.tsv', sep="\t")
#Asset included index as first column, remove it before further processing
full_train_data = full_train_data.iloc[: , 1:]
print("Full Train/Val data shape: ",full_train_data.shape)

### Get trainval features that were generated in previous script
full_test_data = pd.read_csv('/data/tdc_bbb_martins_ml_data/test.tsv', sep="\t")
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
    sm = SVMSMOTE(random_state = 8516, sampling_strategy = {0: int(1.5*sum(df.iloc[:,0] == 1)), 1: sum(df.iloc[:,0] == 1)})
    X_res, y_res = sm.fit_resample(df.iloc[:,1:], df.iloc[:,0].ravel())
    y_res = pd.DataFrame(y_res, columns=['target'])
    resampled_train_data = pd.DataFrame(pd.concat([y_res, X_res], axis=1))
    print("After OverSampling, counts of label '0': {}".format(sum(resampled_train_data.iloc[:,0] == 0)))
    print("After OverSampling, counts of label '1': {} \n".format(sum(resampled_train_data.iloc[:,0] == 1)))
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

with open("/data/tdc_bbb_martins_correlated_features/correlated_features/correlated_fingerprints_to_drop.pkl", "rb") as fp:
    correlated_fingerprints_to_drop = pickle.load(fp)

with open("/data/tdc_bbb_martins_correlated_features/correlated_features/correlated_numeric_to_drop.pkl", "rb") as num_corr:
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

# 7) fit kpca on trainval, transform trainval and test

from sklearn.decomposition import KernelPCA

def kpca_poly_fit(df):
    X=df.iloc[:,1:]
    features = list(X.columns)
    y=pd.DataFrame(df.iloc[:,0])
    kpca = KernelPCA(kernel='poly', random_state=global_seed, remove_zero_eig=True, degree=2, gamma=0.001) # , n_components=500
    #kpca = KernelPCA(kernel='poly', random_state=global_seed, remove_zero_eig=True, degree=5, n_components=500) 
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

#full_train_data_ml_kpca_augmented = augment_data(full_train_data_ml_kpca)

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


def build_best_model(best_params):
    
    
    def create_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        #print("n_layers: ",best_params['n_layers'])
        for i in range(best_params['n_layers']):
            #print("Neurons in Layer {}: ".format(i), best_params['n_units_l{}'.format(i)])
            model.add(tf.keras.layers.Dense(best_params['n_units_l{}'.format(i)], activation='relu', kernel_regularizer=regularizers.l2(best_params['weight_decay'])))
            #print("Dropout after Layer {}: ".format(i), best_params['dropout_l{}'.format(i)])
            model.add(tf.keras.layers.Dropout(best_params['dropout_l{}'.format(i)]))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        #print("Model Created")
        return model
    
    def create_optimizer():
        kwargs = {}
        optimizer_selected = best_params['optimizer']
        if optimizer_selected == "RMSprop":
            kwargs["learning_rate"] = best_params['rmsprop_learning_rate']
            kwargs["weight_decay"] = best_params['rmsprop_weight_decay']
            kwargs["momentum"] =  best_params['rmsprop_momentum']
        elif optimizer_selected == "Adam":
            kwargs["learning_rate"] = best_params['adam_learning_rate']
        elif optimizer_selected == "SGD":
            kwargs["learning_rate"] = best_params['sgd_opt_learning_rate']
            kwargs["momentum"] = best_params['sgd_opt_momentum']

        optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
        #print("Optimizer Created")
        return optimizer
    
    def create_callbacks():
        #create a learning rate reducer search
        from keras.callbacks import ReduceLROnPlateau
        lr_factor = best_params['lr_factor']
        lr_patience = best_params['lr_factor']
        min_lr = best_params['min_lr']
        reduce_lr = ReduceLROnPlateau(mode='auto', factor=lr_factor, patience=lr_patience, min_lr=min_lr, verbose=1)
    
        #create an early stopping callback
        patience = 300
        early_stopping = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.00001,
            patience=patience,
            verbose=1,
            mode='min',
            baseline=None,
            restore_best_weights=True
        )]

        callbacks = [reduce_lr, early_stopping]
        #print("Callbacks Created")
        return callbacks
    
    def sensitivity(y_true, y_pred): 
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def specificity(y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

    def compile_model(model, optimizer):
        model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['AUC', 'Accuracy',sensitivity,specificity])
        #print("Model Compiled")
        return model
        
    # Build model and optimizer.
    built_model = create_model()
    optimizer = create_optimizer()
    model = compile_model(built_model, optimizer)
    callbacks = create_callbacks()
    
    BATCH_SIZE = best_params['BATCH_SIZE']
    
    return model, callbacks, BATCH_SIZE


#### ******* Best from DNN Designer v7 ******* #######
# trial 63 from DNN_designer_RMS_AUC_v7.py (scored {'bbb_martins': [0.912, 0.003]})
trial_params = {'BATCH_SIZE': 57, 'n_layers': 4, 'weight_decay': 1.5973504936266557e-08, 'n_units_l0': 67, 'dropout_l0': 0.028971370414791225, 'activation_l0': 'relu', 'n_units_l1': 70, 'dropout_l1': 0.12208262960893639, 'activation_l1': 'tanh', 'n_units_l2': 84, 'dropout_l2': 0.038719572284664015, 'activation_l2': 'relu', 'n_units_l3': 88, 'dropout_l3': 0.16204810346172394, 'activation_l3': 'relu', 'optimizer': 'RMSprop', 'rmsprop_learning_rate': 0.001, 'rmsprop_weight_decay': 0.954684599370517, 'rmsprop_momentum': 0.013194737611001596, 'lr_factor': 0.4935859743089657, 'lr_patience': 3, 'min_lr': 9.329778269014784e-06}

########################################
#### Picked by DNN Designer v10  #######
########################################

# trial 21 from DNN_designer_RMS_AUC_v10.py (scored {'bbb_martins': [0.901, 0.004]})
#trial_params = {'BATCH_SIZE': 35, 'n_layers': 4, 'weight_decay': 5.611169180553946e-07, 'n_units_l0': 195, 'dropout_l0': 0.03285232604592127, 'activation_l0': 'relu', 'n_units_l1': 61, 'dropout_l1': 0.11154088729037724, 'activation_l1': 'tanh', 'n_units_l2': 167, 'dropout_l2': 0.19793871506797636, 'activation_l2': 'tanh', 'n_units_l3': 58, 'dropout_l3': 0.17659771949649294, 'activation_l3': 'tanh', 'optimizer': 'RMSprop', 'rmsprop_learning_rate': 0.001, 'rmsprop_weight_decay': 0.9688376460078791, 'rmsprop_momentum': 0.002984609972420994, 'lr_factor': 0.5832885199652754, 'lr_patience': 4, 'min_lr': 9.964689122967635e-06}

# trial 23 from DNN_designer_RMS_AUC_v10.py (scored {'bbb_martins': [0.895, 0.007]})
#trial_params = {'BATCH_SIZE': 40, 'n_layers': 4, 'weight_decay': 4.739165848140964e-07, 'n_units_l0': 211, 'dropout_l0': 0.06520070393548646, 'activation_l0': 'relu', 'n_units_l1': 79, 'dropout_l1': 0.09043679689896553, 'activation_l1': 'tanh', 'n_units_l2': 181, 'dropout_l2': 0.19340535068656903, 'activation_l2': 'tanh', 'n_units_l3': 62, 'dropout_l3': 0.18018005423160657, 'activation_l3': 'tanh', 'optimizer': 'RMSprop', 'rmsprop_learning_rate': 0.001, 'rmsprop_weight_decay': 0.9723112061081753, 'rmsprop_momentum': 0.0028735123153180338, 'lr_factor': 0.6326133148278134, 'lr_patience': 4, 'min_lr': 9.944196586213679e-06}

# trial 24 from DNN_designer_RMS_AUC_v10.py (scored {'bbb_martins': [0.901, 0.006]})
#trial_params = {'BATCH_SIZE': 41, 'n_layers': 4, 'weight_decay': 3.939155927069406e-07, 'n_units_l0': 211, 'dropout_l0': 0.06118576791213352, 'activation_l0': 'relu', 'n_units_l1': 77, 'dropout_l1': 0.08953843572720273, 'activation_l1': 'tanh', 'n_units_l2': 217, 'dropout_l2': 0.18130422712405872, 'activation_l2': 'tanh', 'n_units_l3': 67, 'dropout_l3': 0.15171951571844683, 'activation_l3': 'tanh', 'optimizer': 'RMSprop', 'rmsprop_learning_rate': 0.001, 'rmsprop_weight_decay': 0.9732995406174811, 'rmsprop_momentum': 0.002626040588243744, 'lr_factor': 0.6409219398127897, 'lr_patience': 7, 'min_lr': 8.30344195134303e-06}


########################################
#Picked by DNN_designer_RMS_v11_kpca_no_fs.py

# trial 31 from DNN_designer_RMS_v11_kpca_no_fs.py (scored {'bbb_martins': [0.901, 0.006]})
#trial_params = {'BATCH_SIZE': 64, 'n_layers': 4, 'weight_decay': 5.611654640916228e-07, 'n_units_l0': 221, 'dropout_l0': 0.10544785178171406, 'activation_l0': 'relu', 'n_units_l1': 201, 'dropout_l1': 0.0673432592978363, 'activation_l1': 'tanh', 'n_units_l2': 119, 'dropout_l2': 0.11953138030741352, 'activation_l2': 'tanh', 'n_units_l3': 201, 'dropout_l3': 0.16745823408084465, 'activation_l3': 'relu', 'optimizer': 'RMSprop', 'rmsprop_learning_rate': 1e-05, 'rmsprop_weight_decay': 0.9689632251444065, 'rmsprop_momentum': 0.01142186096472865, 'lr_factor': 0.5736197586102744, 'lr_patience': 7, 'min_lr': 2.2802079091962418e-07}


########################################
print("trial params: ", trial_params)
epochs= 10000

# Load data to be used for training
trainval = full_train_data_ml_kpca

    
# Load test data to be used for training
test = full_test_data_ml_kpca

    
loss=[]
AUC=[]
accuracy=[]
sensitivity=[]
specificity=[]
    
### Get the TDC train and test split Drug_IDs to split similarly
group = admet_group(path = 'data/')
predictions_list = []

results_df = pd.DataFrame(columns = ["Model", "Data Set", "AUC", "Accuracy", "f1 Score", "sensitivity", "specificity", "seed"])

for seed in [1, 2, 3, 4, 5]:
    reset_seed(global_seed)
    benchmark = group.get('BBB_Martins')
    name = benchmark['name']
    
    # get the Drug ID's for train/valid/test splits
    train_split, test_split = benchmark['train_val'], benchmark['test']
    train_split, valid_split = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
    train_data_Ids = train_split['Drug_ID']
    valid_data_Ids = valid_split['Drug_ID']
    test_data_Ids = test_split['Drug_ID']
        
    train = full_train_data_ml_kpca[full_train_data_ml_kpca.index.isin(train_data_Ids)]
    validation = full_train_data_ml_kpca[full_train_data_ml_kpca.index.isin(valid_data_Ids)]
    test = full_test_data_ml_kpca[full_test_data_ml_kpca.index.isin(test_data_Ids)]
    
    print("train shape: ", train.shape)
    print("validation shape: ", validation.shape)
    print("test shape: ", test.shape)
    
    train = augment_data(train)
    
    print("train shape after augmenting: ", train.shape)

    
        
    # To use selected features (from non-aug trainval) only
    X_train,y_train, features_train=data_prep(train[['target'] + lasso_features_selected],scale="None")
    X_val,y_val, features_train=data_prep(validation[['target'] + lasso_features_selected],scale="None")
    X_test,y_test, features_test=data_prep(test[['target'] + lasso_features_selected],scale="None")
    print("X_train shape: ",X_train.shape)

    model, callbacks, BATCH_SIZE = build_best_model(trial_params)    
    model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, validation_data=[X_val, y_val], verbose=1, callbacks=callbacks)

    def dnn_predict(trained_model, features):
        model_prediction_prob = trained_model.predict(features, verbose=2)
        model_prediction_class = [1 if pred > 0.5 else 0 for pred in model_prediction_prob]
        model_prediction_prob = [item for sublist in model_prediction_prob for item in sublist]
        return model_prediction_prob, model_prediction_class

    predictions = {}
    name = benchmark['name']

    test_prediction_prob, test_prediction_class = dnn_predict(model, X_test)

    predictions[name] = test_prediction_prob

    predictions_list.append(predictions)

    test_auc = roc_auc_score(test['target'], test_prediction_prob, average='weighted')
    print("\n Test AUC seed {}: ".format(seed),test_auc, "\n")

        # write output files
    # write the above dataframe as .tsv file to use for feature selection
    test_predictions_df = pd.DataFrame({'Drug_ID':test.index.values, 'Actual_value':test.target, 'Predicted_prob':test_prediction_prob, 'Predicted_class':test_prediction_class})
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

    results_df.loc[len(results_df.index)] = ["DNN", "test", test_AUC, test_Accuracy, test_f1, test_sensitivity, test_specificity, seed]

    #Make predictions on the validation set
    val_prediction_prob, val_prediction_class = dnn_predict(model, X_val)

        # write the above dataframe as .tsv file to use for feature selection
    val_predictions_df = pd.DataFrame({'Drug_ID':validation.index.values, 'Actual_value':validation.target, 'Predicted_prob':val_prediction_prob, 'Predicted_class':val_prediction_class})
    val_predictions_df.to_csv("../results/{}/val_predictions/val_predictions_seed{}.tsv".format(method_name,seed), sep="\t", index=0)

    val_AUC, val_Accuracy, val_f1, val_sensitivity, val_specificity = get_metrics(val_predictions_df.Actual_value, val_predictions_df.Predicted_prob, val_predictions_df.Predicted_class)
    print("val AUC: ", val_AUC)
    print("val Accuracy: ",val_Accuracy)
    print("val f1 score: ", val_f1)
    print("sensitivity score: ", val_sensitivity)
    print("specificity score: ", val_specificity)

    results_df.loc[len(results_df.index)] = ["DNN", "validation", val_AUC, val_Accuracy, val_f1, val_sensitivity, val_specificity, seed]

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

#Make out of fold predictions on trainval for use in ensemble
from sklearn.model_selection import cross_val_predict

#### Insert oof code here after model restructure to KerasClassifier which is compatible with cross_val_predict #####

results_df.to_csv("../results/{}/model_performance.tsv".format(method_name), sep="\t", index=0)


