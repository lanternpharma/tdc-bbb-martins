#!/usr/bin/env python
# coding: utf-8

#  Copyright (c) 2023. Lantern Pharma Inc. All rights reserved.

import os

# create output folders for the results
method_name = 'dnn_fe_aug'

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

# import RDKit ----------------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.PandasTools import LoadSDF
import numpy as np
import networkx as nx
from karateclub import Graph2Vec
import pandas as pd
from tdc.single_pred import ADME
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
import random
from tdc.benchmark_group import admet_group
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors

from imblearn.over_sampling import SVMSMOTE

from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit, RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss, confusion_matrix

from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Load LGBM packages with optuna for hyper parameter tuning
import optuna.integration.lightgbm as lgb
import optuna
from lightgbm import early_stopping

import pickle

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

# define the function for coverting rdkit object to networkx object -----------     
def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
        
    return G


# spliting of the scaled data into training (70%) and the testing (30%) - getting dataframe
def split_by_fractions_df(df:pd.DataFrame, fracs:list, random_state:int=42):
    assert sum(fracs)==1.0, 'fractions sum is not 1.0 (fractions_sum={})'.format(sum(fracs))
    remain = df.index.copy().to_frame()
    res = []
    for i in range(len(fracs)):
        fractions_sum=sum(fracs[i:])
        frac = fracs[i]/fractions_sum
        idxs = remain.sample(frac=frac, random_state=random_state).index
        remain=remain.drop(idxs)
        res.append(idxs)
    return [df.loc[idxs] for idxs in res]

# spliting of the scaled data into training (70%) and the testing (30%) - getting index
def split_by_fractions_index(df:pd.DataFrame, fracs:list, random_state:int=42):
    assert sum(fracs)==1.0, 'fractions sum is not 1.0 (fractions_sum={})'.format(sum(fracs))
    remain = df.index.copy().to_frame()
    res = []
    for i in range(len(fracs)):
        fractions_sum=sum(fracs[i:])
        frac = fracs[i]/fractions_sum
        idxs = remain.sample(frac=frac, random_state=random_state).index
        remain=remain.drop(idxs)
        res.append(idxs)
    return [idxs for idxs in res]


# Fingerprint and descriptors functions

def prepare_df(df):
    df.rename(columns={"Y": "target"}, inplace=True)
    df.loc[df["target"] == 0, "target"] = "Non_permeable"
    df.loc[df["target"] == 1, "target"] = "Permeable"
    # print(train_df.shape)
    # print(train_df.groupby('target'). size())
    # train_df.head()
    
    # prepare molecule from drug SMILE structure
    df['mol'] = df['Drug'].apply(lambda x: Chem.MolFromSmiles(x))
    
    # create 2D graph from mol data
    df['graph'] = df['mol'].apply(lambda x: mol_to_nx(x))
    
    # https://www.rdkit.org/docs/GettingStartedInPython.html
    # add 3D graph column
    df['3d_mol'] = df['mol'].apply(lambda x: Chem.AddHs(x))
    
    # create graph from mol data
    df['3d_graph'] = df['3d_mol'].apply(lambda x: mol_to_nx(x))
    
    return(df)


# Calculate the descriptors using 2D structure
# https://towardsdatascience.com/basic-molecular-representation-for-machine-learning-b6be52e9ff76

def desc_2D(df):
    print(">>> create graph embedding ... ")
    random.seed(10)
    model_2d = Graph2Vec()
    model_2d.fit(df['graph'])
    data_smile_2d_graph2vec = model_2d.get_embedding()
    
    data_smile_2d_graph2vec = pd.DataFrame(data_smile_2d_graph2vec)
    
    # rename the columns
    graph_embd_2d_col_list = ['Graph_embd_2d_' + str(x) for x in range(1,data_smile_2d_graph2vec.shape[1] + 1)]
    data_smile_2d_graph2vec.columns = graph_embd_2d_col_list
    
    # print(">>> df_2d_graph2vec shape = ", data_smile_2d_graph2vec.shape)
    # data_smile_2d_graph2vec.head()
    
    # merge with target data
    graph_embd_2d_data = df[['Drug_ID','target']].join(data_smile_2d_graph2vec, how='outer')
    # print(">>> full_data shape = ", graph_embd_2d_data.shape)
    # graph_embd_2d_data.head()
    
    return(graph_embd_2d_data)


# Calculate the descriptors using 3D structure

def desc_3D(df):
    print(">>> create graph embedding ... ")
    random.seed(10)
    model_3d = Graph2Vec()
    model_3d.fit(df['3d_graph'])
    data_smile_3d_graph2vec = model_3d.get_embedding()
    
    data_smile_3d_graph2vec = pd.DataFrame(data_smile_3d_graph2vec)
    
    # rename the columns
    graph_embd_3d_col_list = ['Graph_embd_3d_' + str(x) for x in range(1,data_smile_3d_graph2vec.shape[1] + 1)]
    data_smile_3d_graph2vec.columns = graph_embd_3d_col_list
    
    # merge with target data
    graph_embd_3d_data = df[['Drug_ID','target']].join(data_smile_3d_graph2vec, how='outer')
    # print(">>> full_data shape = ", graph_embd_3d_data.shape)
    # graph_embd_3d_data.head()
    
    return(graph_embd_3d_data)


# Create rdk fingerprint dataframe
def rdk_finger(df):
    
    # it doesnt have chirality option
    def rdk_finger_to_df(mol):
        my_print = RDKFingerprint(mol)
        my_print = np.array(my_print)
        my_df = pd.DataFrame(my_print)
        return my_df
    
    rdk_output_list = [rdk_finger_to_df(i) for i in df['mol']]
    
    fingerprint_rdk_df = pd.DataFrame(list(map(np.ravel, rdk_output_list)))
    
    # rename the columns
    rdk_col_list = ['RDK_' + str(x) for x in range(1,fingerprint_rdk_df.shape[1] + 1)]
    fingerprint_rdk_df.columns = rdk_col_list
    
    # merge with target data
    fingerprint_rdk_data = df[['Drug_ID','target']].join(fingerprint_rdk_df, how='outer')
    # print(">>> full_data shape = ", fingerprint_rdk_data.shape)
    # fingerprint_rdk_data.head()
    
    return(fingerprint_rdk_data)


# Create morgan fingerprint dataframe
def morgan_finger(df):
    
    # it has chirality option
    def morgan_finger_to_df(mol):
        my_print = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2)
        my_print = np.array(my_print)
        my_df = pd.DataFrame(my_print)
        return my_df

    # 150 chemical structure 150th can show difference in fingerprints if we disable the chairality
    # mol = data_smile['mol'][150]
    # my_print = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2)
    # my_print = np.array(my_print)
    # my_print[1:100]
    
    morgan_output_list = [morgan_finger_to_df(i) for i in df['mol']]
    
    fingerprint_morgan_df = pd.DataFrame(list(map(np.ravel, morgan_output_list)))
    
    # rename the columns
    morgan_col_list = ['Morgan_' + str(x) for x in range(1,fingerprint_morgan_df.shape[1] + 1)]
    fingerprint_morgan_df.columns = morgan_col_list
    
    # merge with target data
    fingerprint_morgan_data = df[['Drug_ID','target']].join(fingerprint_morgan_df, how='outer')
    # print(">>> full_data shape = ", fingerprint_morgan_data.shape)
    # fingerprint_morgan_data.head()
    
    return(fingerprint_morgan_data)


# Create MACCS fingerprint dataframe
def maccs_finger(df):
    # it doesnt have chirality option
    def maccs_finger_to_df(mol):
        my_print = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        my_print = np.array(my_print)
        my_df = pd.DataFrame(my_print)
        return my_df
    
    maccs_output_list = [maccs_finger_to_df(i) for i in df['mol']]
    
    fingerprint_maccs_df = pd.DataFrame(list(map(np.ravel, maccs_output_list)))
    
    # rename the columns
    maccs_col_list = ['MACCS_' + str(x) for x in range(1,fingerprint_maccs_df.shape[1] + 1)]
    fingerprint_maccs_df.columns = maccs_col_list
    
    # merge with target data
    fingerprint_maccs_data = df[['Drug_ID','target']].join(fingerprint_maccs_df, how='outer')
    # print(">>> full_data shape = ", fingerprint_maccs_data.shape)
    # fingerprint_maccs_data.head()
    
    return(fingerprint_maccs_data)

# Calculate 2D Autocorrelation descriptors vector from 2d molecule
def autocor_2d(df):
    def autocor_2D_desc_to_df(mol):
        my_print = Chem.rdMolDescriptors.CalcAUTOCORR2D(mol)
        my_print = np.array(my_print)
        my_df = pd.DataFrame(my_print)
        return my_df

    autocor_2D_desc_list = [autocor_2D_desc_to_df(i) for i in df['mol']]
    
    autocor_2D_desc_df = pd.DataFrame(list(map(np.ravel, autocor_2D_desc_list)))
    
    # rename the columns
    autocor_2D_col_list = ['Autocor2D_' + str(x) for x in range(1,autocor_2D_desc_df.shape[1] + 1)]
    autocor_2D_desc_df.columns = autocor_2D_col_list
    
    # merge with target data
    autocorr_2d_desc_data = df[['Drug_ID','target']].join(autocor_2D_desc_df, how='outer')
    # print(">>> full_data shape = ", autocorr_2d_desc_data.shape)
    # autocorr_2d_desc_data.head()
    
    return(autocorr_2d_desc_data)


# Calculate 3D Autocorrelation descriptors vector from 3d molecule
def autocor_3d(df):
    # ref = https://github.com/rdkit/UGM_2017/blob/master/Presentations/Godin_3D_Descriptors.pdf
    # ref = https://github.com/rdkit/rdkit/issues/2924
    # Returns 3D Autocorrelation descriptors vector
    # Chem.rdMolDescriptors.CalcAUTOCORR3D(data_smile['mol'][1])
    
    df['3d_mol'].apply(lambda x: AllChem.EmbedMolecule(x))
    
    # temp_vec = Chem.rdMolDescriptors.CalcAUTOCORR3D(data_smile['3d_mol'][1977])

    def autocor_3D_desc(mol):
        try:
            temp_vec = Chem.rdMolDescriptors.CalcAUTOCORR3D(mol)
            return temp_vec
        except:
            # print("not able to calculate for row:", i , "and drugname is:",data_smile['Drug_ID'][i])
            temp_vec_0 = [0] * 80
            return temp_vec_0
            
    # convert list into dataframe
    autocor_3D_desc_list = [autocor_3D_desc(i) for i in df['3d_mol']]
    autocor_3D_desc_df = pd.DataFrame(autocor_3D_desc_list)
    
    # rename the columns
    autocor_3D_col_list = ['Autocor3D_' + str(x) for x in range(1,autocor_3D_desc_df.shape[1] + 1)]
    autocor_3D_desc_df.columns = autocor_3D_col_list
    
    # merge with target data
    autocorr_3d_desc_data = df[['Drug_ID','target']].join(autocor_3D_desc_df, how='outer')
    # print(">>> full_data shape = ", autocorr_3d_desc_data.shape)
    # autocorr_3d_desc_data.head()
    
    return(autocorr_3d_desc_data)

def multi_rule_desc_df(df):

    # calculate the descriptors from multiple rules
    def multi_rule_desc_cal(mol):
        molecular_weight = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_bond_donor = Descriptors.NumHDonors(mol)
        h_bond_acceptors = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        number_of_atoms = Chem.rdchem.Mol.GetNumAtoms(mol)
        molar_refractivity = Chem.Crippen.MolMR(mol)
        topological_surface_area_mapping = Chem.QED.properties(mol).PSA
        formal_charge = Chem.rdmolops.GetFormalCharge(mol)
        heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(mol)
        num_of_rings = Chem.rdMolDescriptors.CalcNumRings(mol)

        molecular_weight = np.array(molecular_weight)
        logp = np.array(logp)
        h_bond_donor = np.array(h_bond_donor)
        h_bond_acceptors = np.array(h_bond_acceptors)
        rotatable_bonds = np.array(rotatable_bonds)
        number_of_atoms = np.array(number_of_atoms)
        molar_refractivity = np.array(molar_refractivity)
        topological_surface_area_mapping = np.array(topological_surface_area_mapping)
        formal_charge = np.array(formal_charge)
        heavy_atoms = np.array(heavy_atoms)
        num_of_rings = np.array(num_of_rings)

        return molecular_weight, logp, h_bond_donor, h_bond_acceptors, rotatable_bonds, number_of_atoms, molar_refractivity, topological_surface_area_mapping, formal_charge, heavy_atoms, num_of_rings

    multi_rule_desc_list = [multi_rule_desc_cal(i) for i in df['mol']]
    multi_rule_desc_df = pd.DataFrame(list(map(np.ravel, multi_rule_desc_list)))

    # rename the columns
    multi_rule_desc_df_col =('molecular_weight', 'logp', 'h_bond_donor', 'h_bond_acceptors', 'rotatable_bonds', 
                             'number_of_atoms', 'molar_refractivity', 'topological_surface_area_mapping', 
                             'formal_charge', 'heavy_atoms', 'num_of_rings')
    multi_rule_desc_df.columns = multi_rule_desc_df_col

    multi_rule_desc_data = df[['Drug_ID','target']].join(multi_rule_desc_df, how='outer')

    return(multi_rule_desc_data)
    # return(multi_rule_desc_list)


# Druglikeness is a qualitative concept used in drug design for how "druglike" a substance is with respect to factors like bioavailability
# Lipinski’s “rule of five” relates BBB permeability to molecular weight, lipophilicity, polar surface area, hydrogen bonding, and charge - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4064947/

def merge_rules(df):
    
    # lipinski rule check
    def lipinski_rule(s):
        if (s['molecular_weight'] <= 500) and (s['logp'] <= 5) and (s['h_bond_donor'] <= 5) and (s['h_bond_acceptors'] <= 5) and (s['rotatable_bonds'] <= 5):
            return 1
        else:
            return 0
    df['lipinski_rule'] = df.apply(lipinski_rule, axis=1)
    
    # Ghose Filter
    # molecular_weight >= 160 and molecular_weight <= 480 and logp >= 0.4 and logp <= 5.6 and number_of_atoms >= 20 and number_of_atoms <= 70 and molar_refractivity >= 40 and molar_refractivity <= 130:
    def ghose_filter(s):
        if (s['molecular_weight'] >= 160) and (s['molecular_weight'] <= 480) and (s['logp'] >= 0.4) and (s['logp'] <= 5.6) and (s['number_of_atoms'] >= 20) and (s['number_of_atoms'] <= 70) and (s['molar_refractivity'] >= 40) and (s['molar_refractivity'] <= 130):
            return 1
        else:
            return 0
    df['ghose_filter'] = df.apply(ghose_filter, axis=1)
    
    # veber Filter
    # if rotatable_bonds <= 10 and topological_surface_area_mapping <= 140:
    def veber_filter(s):
        if (s['rotatable_bonds'] <= 10) and (s['topological_surface_area_mapping'] <= 140):
            return 1
        else:
            return 0
    df['veber_filter'] = df.apply(veber_filter, axis=1)
    
    # rule of 3
    # molecular_weight <= 300 and logp <= 3 and h_bond_donor <= 3 and h_bond_acceptors <= 3 and rotatable_bonds <= 3:
    def rule_of_3(s):
        if (s['molecular_weight'] <= 300) and (s['logp'] <= 3) and (s['h_bond_donor'] <= 3) and (s['h_bond_acceptors'] <= 3) and (s['rotatable_bonds'] <= 3):
            return 1
        else:
            return 0
    df['rule_of_3'] = df.apply(rule_of_3, axis=1)
    
    # REOS Filter
    # molecular_weight >= 200 and molecular_weight <= 500 and logp >= int(0 - 5) and logp <= 5 and h_bond_donor >= 0 and h_bond_donor <= 5 and h_bond_acceptors >= 0 and h_bond_acceptors <= 10 and formal_charge >= int(0-2) and formal_charge <= 2 and rotatable_bonds >= 0 and rotatable_bonds <= 8 and heavy_atoms >= 15 and heavy_atoms <= 50:
    def REOS_filter(s):
        if (s['molecular_weight'] >= 200) and (s['molecular_weight'] <= 500) and (s['logp'] >= int(0 - 5)) and (s['logp'] <= 5) and (s['h_bond_donor'] >= 0) and (s['h_bond_donor'] <= 5) and (s['h_bond_acceptors'] >= 0) and (s['h_bond_acceptors'] <= 10) and (s['formal_charge'] >= int(0-2)) and (s['formal_charge'] <= 2) and (s['rotatable_bonds'] >= 0) and (s['rotatable_bonds'] <= 8) and (s['heavy_atoms'] >= 15) and (s['heavy_atoms'] <= 50):
            return 1
        else:
            return 0
    df['REOS_filter'] = df.apply(REOS_filter, axis=1)
    
    # Drug Like Filter
    # if molecular_weight < 400 and num_of_rings > 0 and rotatable_bonds < 5 and h_bond_donor <= 5 and h_bond_acceptors <= 10 and logp < 5:
    def drug_like(s):
        if (s['molecular_weight'] < 400) and (s['num_of_rings'] > 0) and (s['rotatable_bonds'] < 5) and (s['h_bond_donor'] <= 5) and (s['h_bond_acceptors'] <= 10) and (s['logp'] < 5):
            return 1
        else:
            return 0
    df['drug_like'] = df.apply(drug_like, axis=1)
    
    return df

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

    neg_sig_fingerprints = list(chi2_fingerprints_df[(chi2_fingerprints_df.percent_permeable < 0.4) & (chi2_fingerprints_df.p_value < 0.05)].Feature)

    pos_sig_fingerprints = list(chi2_fingerprints_df[(chi2_fingerprints_df.percent_permeable > 0.80) & (chi2_fingerprints_df.p_value < 0.05)].Feature)
    return neg_sig_fingerprints, pos_sig_fingerprints

# Create funtion to take sig fingerprint lists and add features for count of sig fingerprints

def get_num_fingerprints(df, neg_sig_fingerprints, pos_sig_fingerprints):
    df['neg_sig_fingerprints'] = df[neg_sig_fingerprints].sum(axis=1)
    df['pos_sig_fingerprints'] = df[pos_sig_fingerprints].sum(axis=1)
    return df

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


#Add function for Logistic L1 feature selection

def logistic_selection(df, scale_method):
    X,y,features=data_prep(df,scale=scale_method)
    # Perform GridSearchCV to tune best-fit LR model
    param = {
        'C': np.arange(0, 1, 0.01), #np.arange(0, 1, 0.01)
        'class_weight': ['None', 'balanced'], #['None', 'balanced']
        'solver': ['liblinear'], #['liblinear', 'saga']
        'random_state': [global_seed]
    }
    lr_model = LogisticRegression(penalty='l1')
    gs_model = GridSearchCV(estimator=lr_model, param_grid=param, cv=10)
    gs_model.fit(X, y)
    print("Best Parameters from search: ",gs_model.best_params_)
    # Train a LR model with best parameters
    model = LogisticRegression(**gs_model.best_params_, penalty='l1')
    model.fit(X, y)
    coef = model.coef_[0]
    rankings = pd.DataFrame(list(zip(features, coef)), columns=['feature','coef'])
    rankings.sort_values(by='coef', ascending=False, key=abs, inplace=True)
    rankings = rankings[rankings.coef!=0]
    rankings['Rank'] = range(1,len(rankings.feature)+1,1)
    features_selected = list(rankings.feature)
    return features_selected, rankings
    
# Hyperband search for optimal neural network design and parameters
from tensorflow.keras import backend as K

# function to train deep neural network model

def train_dnn(train_df, val_df, scale_method):
    #split features and targets for modeling
    X_train,y_train, features_train=data_prep(train_df,scale=scale_method)
    X_val,y_val, features_val=data_prep(val_df,scale=scale_method)

    #Set Max number of epochs to 1 for code debugging, then increase to 200 when ready for modeling
    max_epochs = 200
    
    #Add early stopping
    patience = 100
    callback = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00001,
        patience=patience,
        verbose=3,
        mode='min',
        baseline=None,
        restore_best_weights=True
    )]
    
    # Define fixed parameters
    BATCH_SIZE = 16
    dropout=0.3
    
    # Define search space for design of model architecture and hyperparameters
    def build_model(hp):
        model = tf.keras.Sequential()
        for i in range(hp.Int('layers', 4, 5)):
            model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 128, 400, step=128),
                                    activation=hp.Choice('act_' + str(i), ['relu']), kernel_regularizer=regularizers.l2(0.01)))
            model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer = 'adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy', 'AUC'])
        return model
    
    # Define Hyperband search based on grid defined above
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=max_epochs,
        directory='keras_tuner_dir',
        project_name='keras_tuner_demo',
        overwrite=True,
        seed=global_seed,
    )
    
    # Output summary of search space
    tuner.search_space_summary()
    
    #Run Hyperband search for optimal design of neural network model
    tuner.search(X_train, y_train, epochs=max_epochs, batch_size=BATCH_SIZE, validation_data=[X_val, y_val], callbacks=[callback],verbose=0)

    # Collect best model from Hyperband search
    best_model_from_search = tuner.get_best_models()[0]

    # Re-train best model from Hyperband to early stopping criteria
    search_model_retrained = best_model_from_search
    search_model_retrained.compile(optimizer = 'adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy', 'AUC'])
    history = History()
    history = search_model_retrained.fit(X_train, y_train, epochs=max_epochs, batch_size=BATCH_SIZE, validation_data=[X_val, y_val], verbose=2, callbacks=callback)
    search_model_retrained.summary()

    return search_model_retrained


# Function to make predictions 

def dnn_predict(trained_model, df, scale_method):
    #split features and targets for modeling
    X_test,y_test, features_test=data_prep(df,scale=scale_method)
    model_prediction_prob = trained_model.predict(X_test, verbose=2)
    model_prediction_class = [1 if pred > 0.5 else 0 for pred in model_prediction_prob]
    model_prediction_prob = [item for sublist in model_prediction_prob for item in sublist]
    return model_prediction_prob, model_prediction_class


# Get the TDC full train, val, test data with no splits to use for feature generation

group = admet_group(path = 'data/')
benchmark = group.get('BBB_Martins')
name = benchmark['name']

train_df, test_df = benchmark['train_val'], benchmark['test']

import datetime
now = datetime.datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H:%M:%S")
print ("TDC train/test data loaded at : ", time_stamp)

# prepare the training/val data for functions used to generate features
full_train_df = prepare_df(train_df)
print(full_train_df.shape)

# apply fingerprint and descriptor's functions on training/val data
fingerprint_rdk_full_train = rdk_finger(full_train_df)
fingerprint_morgan_full_train = morgan_finger(full_train_df)
fingerprint_maccs_full_train = maccs_finger(full_train_df)
autocorr_2d_desc_full_train = autocor_2d(full_train_df)
autocorr_3d_desc_full_train = autocor_3d(full_train_df)
multi_rule_desc_full_train = multi_rule_desc_df(full_train_df)
multi_rule_desc_full_train = merge_rules(multi_rule_desc_full_train)

# merge the data
full_train_data = pd.concat([
                             fingerprint_rdk_full_train, 
                             fingerprint_morgan_full_train.drop(['Drug_ID', 'target'], axis=1), 
                             fingerprint_maccs_full_train.drop(['Drug_ID', 'target'], axis=1), 
                             autocorr_2d_desc_full_train.drop(['Drug_ID', 'target'], axis=1),
                             autocorr_3d_desc_full_train.drop(['Drug_ID', 'target'], axis=1), 
                             multi_rule_desc_full_train.drop(['Drug_ID', 'target'], axis=1)], axis=1)

print("Full Train/Val data shape: ",full_train_data.shape)


import datetime
now = datetime.datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H:%M:%S")
print ("Train feature creation completed at : ", time_stamp)

# prepare the test data for functions used to generate features
test_df = prepare_df(test_df)
print(test_df.shape)

# apply fingerprint and descriptor's functions on training/val data
fingerprint_rdk_test = rdk_finger(test_df)
fingerprint_morgan_test = morgan_finger(test_df)
fingerprint_maccs_test = maccs_finger(test_df)
autocorr_2d_desc_test = autocor_2d(test_df)
autocorr_3d_desc_test = autocor_3d(test_df)
multi_rule_desc_full_test = multi_rule_desc_df(test_df)
multi_rule_desc_full_test = merge_rules(multi_rule_desc_full_test)

# merge the data
full_test_data = pd.concat([
                             fingerprint_rdk_test, 
                             fingerprint_morgan_test.drop(['Drug_ID', 'target'], axis=1), 
                             fingerprint_maccs_test.drop(['Drug_ID', 'target'], axis=1), 
                             autocorr_2d_desc_test.drop(['Drug_ID', 'target'], axis=1),
                             autocorr_3d_desc_test.drop(['Drug_ID', 'target'], axis=1), 
                             multi_rule_desc_full_test.drop(['Drug_ID', 'target'], axis=1)], axis=1)

print("Full test data shape: ",full_test_data.shape)

import datetime
now = datetime.datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H:%M:%S")
print ("Test feature creation completed at : ", time_stamp)

# Get just the train split to use for feature selection
data = ADME(name = 'BBB_Martins')
split = data.get_split()
train_only_df = split['train']
print("Train only shape: ",train_only_df.shape)
train_only_Ids = train_only_df['Drug_ID']

# prepare the training/val data for functions used to generate features
train_only_data = prepare_df(train_only_df)
print(train_only_data.shape)

# apply fingerprint and descriptor's functions on training/val data
fingerprint_rdk_train_only = rdk_finger(train_only_data)
fingerprint_morgan_train_only = morgan_finger(train_only_data)
fingerprint_maccs_train_only = maccs_finger(train_only_data)
autocorr_2d_desc_train_only = autocor_2d(train_only_data)
autocorr_3d_desc_train_only = autocor_3d(train_only_data)
multi_rule_desc_train_only = multi_rule_desc_df(train_only_data)
multi_rule_desc_train_only = merge_rules(multi_rule_desc_train_only)

# merge the data
train_only_for_fs = pd.concat([
                             fingerprint_rdk_train_only, 
                             fingerprint_morgan_train_only.drop(['Drug_ID', 'target'], axis=1), 
                             fingerprint_maccs_train_only.drop(['Drug_ID', 'target'], axis=1), 
                             autocorr_2d_desc_train_only.drop(['Drug_ID', 'target'], axis=1),
                             autocorr_3d_desc_train_only.drop(['Drug_ID', 'target'], axis=1),
                             multi_rule_desc_train_only.drop(['Drug_ID', 'target'], axis=1)], axis=1)

neg_sig_fingerprints, pos_sig_fingerprints = get_fingerprint_lists(train_only_for_fs)
train_only_for_fs = get_num_fingerprints(train_only_for_fs, neg_sig_fingerprints, pos_sig_fingerprints)
full_train_data = get_num_fingerprints(full_train_data, neg_sig_fingerprints, pos_sig_fingerprints)
full_test_data = get_num_fingerprints(full_test_data, neg_sig_fingerprints, pos_sig_fingerprints)

print("Train only for FS shape: ",train_only_for_fs.shape)

#Format the dataframe to ML format for feature selection
train_data_fs_ml, target_dictionary = format_df_to_ml(train_only_for_fs)
print("train_data_fs_ml shape: ",train_data_fs_ml.shape)

#Augment the training data for feature selection
train_data_fs_ml_augmented = augment_data(train_data_fs_ml)

# Run feature selection on training data for each seed in loop
logistic_features_selected, logistic_fs_ranks = logistic_selection(train_data_fs_ml_augmented, "StandardScaler")
print(len(logistic_features_selected))
print(logistic_fs_ranks.head())
logistic_fs_ranks.to_csv("../results/{}/feature_selection/logistic_fs_ranks.tsv".format(method_name), sep="\t", index=0)

### Get the TDC train and test split Drug_IDs to split similarly
group = admet_group(path = 'data/')
predictions_list = []

### use the code provided in the TDC website
#### https://tdcommons.ai/benchmark/overview/

for seed in [1, 2, 3, 4, 5]:
    benchmark = group.get('BBB_Martins')
    name = benchmark['name']
    
    # get the Drug ID's for train/valid/test splits
    train_split, test_split = benchmark['train_val'], benchmark['test']
    train_split, valid_split = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
    train_data_Ids = train_split['Drug_ID']
    valid_data_Ids = valid_split['Drug_ID']
    test_data_Ids = test_split['Drug_ID']
    
    train_df = full_train_data[full_train_data.Drug_ID.isin(train_data_Ids)]
    valid_df = full_train_data[full_train_data.Drug_ID.isin(valid_data_Ids)]
    test_df = full_test_data[full_test_data.Drug_ID.isin(test_data_Ids)]
    
    #Format the dataframe for ML
    full_train_data_ml, target_dictionary = format_df_to_ml(train_df)
    full_valid_data_ml, target_dictionary = format_df_to_ml(valid_df)
    full_test_data_ml, target_dictionary = format_df_to_ml(test_df)
    
    #Augment the training data (not the test or validation data)
    full_train_data_ml_augmented = augment_data(full_train_data_ml)
    
    #Train the model with augmented data, using augmented validation for early stopping
    trained_model = train_dnn(full_train_data_ml_augmented[['target'] + logistic_features_selected], full_valid_data_ml[['target'] + logistic_features_selected], "None")
    
    #Make predictions on the test set
    predictions = {}
    name = benchmark['name']
    predicted_prob, predicted_class = dnn_predict(trained_model, full_test_data_ml[['target'] + logistic_features_selected], "None")
    predictions[name] = predicted_prob

    predictions_list.append(predictions)
    
    # write output files
    # write the above dataframe as .tsv file to use for feature selection
    test_predictions_df = pd.DataFrame({'Drug_ID':full_test_data_ml.index.values, 'Actual_value':full_test_data_ml.target, 'Predicted_prob':predicted_prob, 'Predicted_class':predicted_class})
    test_predictions_df.to_csv("../results/{}/test_predictions/test_predictions_seed{}.tsv".format(method_name, seed), sep="\t", index=0)

    #Make predictions on the validation set
    val_predicted_prob, val_predicted_class = dnn_predict(trained_model, full_valid_data_ml[['target'] + logistic_features_selected], "None")
    # write the above dataframe as .tsv file 
    val_predictions_df = pd.DataFrame({'Drug_ID':full_valid_data_ml.index.values, 'Actual_value':full_valid_data_ml.target, 'Predicted_prob':val_predicted_prob, 'Predicted_class':val_predicted_class})
    val_predictions_df.to_csv("../results/{}/val_predictions/val_predictions_seed{}.tsv".format(method_name,seed), sep="\t", index=0)

    test_auc = roc_auc_score(full_test_data_ml['target'], predicted_prob, average='weighted')
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
