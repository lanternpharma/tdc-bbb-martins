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
method_name = 'correlated_features'

results_path = r'/data/{}/'.format(method_name)
if not os.path.exists(results_path):
    os.makedirs(results_path)

input_path = r'/data/tdc_bbb_martins_ml_data/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# 2) Load data created by generate_features.py

### Get trainval features that were generated in previous script
full_train_data = pd.read_csv('{}trainval.tsv'.format(input_path), sep="\t")
#Asset included index as first column, remove it before further processing
full_train_data = full_train_data.iloc[: , 1:]
print("Full Train/Val data shape: ",full_train_data.shape)

### Get trainval features that were generated in previous script
full_test_data = pd.read_csv('{}test.tsv'.format(input_path), sep="\t")
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
    #print("Before OverSampling, counts of label '0': {}".format(sum(df.iloc[:,0] == 0)))
    #print("Before OverSampling, counts of label '1': {} \n".format(sum(df.iloc[:,0] == 1)))
    sm = SVMSMOTE(random_state = 8516)
    X_res, y_res = sm.fit_resample(df.iloc[:,1:], df.iloc[:,0].ravel())
    y_res = pd.DataFrame(y_res, columns=['target'])
    resampled_train_data = pd.DataFrame(pd.concat([y_res, X_res], axis=1))
    #print("After OverSampling, counts of label '0': {}".format(sum(resampled_train_data.iloc[:,0] == 0)))
    #print("After OverSampling, counts of label '1': {} \n".format(sum(resampled_train_data.iloc[:,0] == 1)))
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
    dump(scaler, open('{}scaler.pkl'.format(results_path), 'wb'))
    

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
#neg_sig_fingerprints_df.to_csv("../results/{}/neg_sig_fingerprints.tsv".format(method_name), sep="\t", index=0)

pos_sig_fingerprints_df = pd.DataFrame(pos_sig_fingerprints, columns=['pos_sig_fingerprints'])
#pos_sig_fingerprints_df.to_csv("../results/{}/pos_sig_fingerprints.tsv".format(method_name), sep="\t", index=0)

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
full_train_data_ml_scaled = scaler_transform(full_train_data_ml, '{}scaler.pkl'.format(results_path))
full_test_data_ml_scaled = scaler_transform(full_test_data_ml, '{}scaler.pkl'.format(results_path)) 

full_train_data_ml_scaled.head()

# 6) Drop highly correlated features (tanimoto or R > 0.95)

# To speed up during development used pre-calculated correlated fingerprints below
#with open("/data/fp_to_drop_point95.pkl", "rb") as fp:
#    fp_to_drop_point95 = pickle.load(fp)

# Calculate highly correlated fingerprints to drop
# Note this needs to be done on the non-scaled set so that fingerprint values are 0/1

rdk_names = list(filter(lambda k: 'RDK' in k, full_train_data_ml.columns))
morgan_names = list(filter(lambda k: 'Morgan' in k, full_train_data_ml.columns))
maccs_names = list(filter(lambda k: 'MACC' in k, full_train_data_ml.columns))
fp_names = rdk_names + morgan_names + maccs_names

# Create dataframe with fingerprints only
trainval_prints = full_train_data_ml[fp_names].copy()

#Transpose for row based calculations
trainval_prints_T = trainval_prints.T

# Create array of fingerprint values across all samples
trainval_prints_T['fp_array'] = trainval_prints_T.values.tolist()
trainval_prints_T = trainval_prints_T[['fp_array']]
trainval_prints_T.fp_array = np.array(trainval_prints_T.fp_array)
import itertools

# create list of all fp-fp combinations to calculate correlation matrix on
fp_pair_list = list(itertools.combinations(fp_names,2))

# join fp1 & fp2 values onto combination id's
fp_pairs_df = pd.DataFrame(fp_pair_list, columns=['fp_name_1', 'fp_name_2'])
fp_pairs_df = fp_pairs_df.merge(trainval_prints_T, how='left', left_on='fp_name_1', right_on=trainval_prints_T.index)
fp_pairs_df.rename({'fp_array':'fp_array_1'}, axis=1, inplace=True)
fp_pairs_df = fp_pairs_df.merge(trainval_prints_T, how='left', left_on='fp_name_2', right_on=trainval_prints_T.index)
fp_pairs_df.rename({'fp_array':'fp_array_2'}, axis=1, inplace=True)

#Now that they are joined, unnest the columns and convert to numpy matrix for faster calculations
def expand_column(column):
    # change each row to int8 numpy array to overcome memory size limitations
    column = column.apply(lambda x: np.array(x, dtype=np.int8))
    # convert from column to numpy array of arrays to reduce memory needed
    array = column.to_numpy()
    # stack into single matrix
    matrix = np.stack(array)
    return matrix

fp1_numpy = expand_column(fp_pairs_df.fp_array_1)
fp2_numpy = expand_column(fp_pairs_df.fp_array_2)

# This operation is memory intense, import garbage collector and delete temp tables throughout process
import gc

def only_one_fp_sum(A, B):
    #print("A: ", A)
    #print("B: ", B)
    A_minus_B = A - B
    #print("A - B: ", A_minus_B)
    only_A = np.where(A_minus_B<0, 0, A_minus_B)
    del A_minus_B
    gc.collect()
    #print("Only in A: ",only_A)
    only_A_sum = np.sum(only_A, axis=1, dtype=np.int16)
    #print("only_A_sum: ",only_A_sum)
    return only_A_sum

def in_both_fp(A, B):
    A_plus_B = A + B
    A_plus_B = np.array(A_plus_B, dtype=np.int8)
    in_A_and_B = np.where(A_plus_B<2, 0, A_plus_B)
    del A_plus_B
    gc.collect()
    in_A_and_B = np.where(in_A_and_B==2, 1, in_A_and_B)
    in_A_and_B = np.array(in_A_and_B, dtype=np.int8)
    sum_in_both = np.sum(in_A_and_B, axis=1, dtype=np.int16)
    del in_A_and_B
    gc.collect()
    return sum_in_both

# tanimoto = bothAB / (onlyA + onlyB + bothAB)
def tanimoto_sim(A,B):
    only_fp_A = only_one_fp_sum(A,B)
    print("Tan function Only in A: ", only_fp_A)
    only_fp_B = only_one_fp_sum(B,A)
    print("Tan function Only in B: ", only_fp_B)
    sum_in_both_fp = in_both_fp(A, B)
    print("Sum in both FP: ", sum_in_both_fp)
    #onlyA_plus_onlyB = np.sum(only_fp_A, only_fp_B, axis=1)
    #denominator = np.sum(onlyA_plus_onlyB, in_both_fp, axis=1)
    denominator = only_fp_A + only_fp_B + sum_in_both_fp
    denominator = np.where(denominator==0, 1, denominator) #replace zeros with 1 so as not to cause division error in next step
    denominator = np.array(denominator, dtype=np.int16)
    print("denomiator: ", denominator)
    tanimoto = np.divide(sum_in_both_fp, denominator)
    print("tanimoto: ", tanimoto)
    return tanimoto

tanimoto_all = tanimoto_sim(fp1_numpy, fp2_numpy)

fp_pairs_df['tanimoto_sim'] = tanimoto_all

tanimoto_pivot = fp_pairs_df.copy().pivot(index='fp_name_1', columns='fp_name_2', values='tanimoto_sim')
tanimoto_pivot_df = pd.DataFrame(tanimoto_pivot.copy())
tanimoto_pivot_df[fp_names[1:]].reindex(fp_names)

upper = tanimoto_pivot_df[fp_names[1:]].reindex(fp_names)

correlated_fingerprints = [column for column in upper.columns if any(upper[column] > 0.95)]
print("# Correlated Fingerprints to drop: ",len(correlated_fingerprints))
print("Fingerprints to drop: ",*correlated_fingerprints)

import pickle
with open("{}correlated_fingerprints_to_drop.pkl".format(results_path), "wb") as fp:
    pickle.dump(correlated_fingerprints, fp)


# Now Create list of correlated numeric features using scaled data generated by previous script

print("full_train_data_ml_scaled columns: ",len(full_train_data_ml_scaled.columns))

full_train_data_ml_scaled.drop(correlated_fingerprints , axis=1, inplace=True)
print("Full Train/Val data shape after dropping correlated fingerprints: ",full_train_data_ml_scaled.shape)

full_test_data_ml_scaled.drop(correlated_fingerprints , axis=1, inplace=True)
print("Full Test data shape after dropping correlated fingerprints: ",full_test_data_ml_scaled.shape)

# Calculate Pearson for non-fingerprint features drop if > 0.95
rdk_names = list(filter(lambda k: 'RDK' in k, full_train_data_ml_scaled.columns))
morgan_names = list(filter(lambda k: 'Morgan' in k, full_train_data_ml_scaled.columns))
maccs_names = list(filter(lambda k: 'MACC' in k, full_train_data_ml_scaled.columns))
fp_names = rdk_names + morgan_names + maccs_names

full_train_data_ml_scaled_non_fp = full_train_data_ml_scaled.drop(fp_names, axis=1).copy()

non_fp_corr_matrix = full_train_data_ml_scaled_non_fp.drop('target',axis=1).corr(method ='pearson')

#just get the upper triangle
upper = np.triu(non_fp_corr_matrix, k=1)
upper = np.where(upper==0, np.nan, upper)
upper = pd.DataFrame(upper, columns=non_fp_corr_matrix.columns, index=non_fp_corr_matrix.index)

correlated_numeric_features = [column for column in upper.columns if any((upper[column] > 0.95) | (upper[column] < -0.95))]
print("# Correlated Numeric Features to drop: ",len(correlated_numeric_features))
print("Correlated Numeric Features to drop: ",*correlated_numeric_features)

full_train_data_ml_scaled.drop(correlated_numeric_features, axis=1, inplace=True)
full_test_data_ml_scaled.drop(correlated_numeric_features, axis=1, inplace=True)
print("Full Train/Val data shape after dropping correlated numeric features: ",full_train_data_ml_scaled.shape)
print("Full Test data shape after dropping correlated numeric features: ",full_test_data_ml_scaled.shape)

with open("{}correlated_numeric_to_drop.pkl".format(results_path), "wb") as num_corr:
    pickle.dump(correlated_numeric_features, num_corr)

