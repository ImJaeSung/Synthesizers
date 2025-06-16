import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)   
    
def generalization_score(train, syndata, continuous_features, categorical_features):
    ### pre-processing
    train_ = train.copy()
    syndata_ = syndata.copy()
    # continuous: standardization
    scaler = StandardScaler().fit(train_[continuous_features])
    train_[continuous_features] = scaler.transform(train_[continuous_features])
    syndata_[continuous_features] = scaler.transform(syndata_[continuous_features])
    # categorical: one-hot encoding
    scaler = OneHotEncoder(handle_unknown='ignore').fit(train_[categorical_features])
    train_ = np.concatenate([
        train_[continuous_features].values,
        scaler.transform(train_[categorical_features]).toarray()
    ], axis=1)
    syndata_ = np.concatenate([
        syndata_[continuous_features].values,
        scaler.transform(syndata_[categorical_features]).toarray()
    ], axis=1)

    allpairs = cosine_similarity(syndata_, train_)
    cos = allpairs.max(axis=1)
    
    return cos
#%%
def DCR_values(train, syndata, continuous_features, categorical_features):
    ### pre-processing
    train_ = train.copy()
    syndata_ = syndata.copy()
    # continuous: min-max scaling
    scaler = MinMaxScaler().fit(train_[continuous_features])
    train_[continuous_features] = scaler.transform(train_[continuous_features])
    syndata_[continuous_features] = scaler.transform(syndata_[continuous_features])
    # categorical: one-hot encoding
    scaler = OneHotEncoder(handle_unknown='ignore').fit(train_[categorical_features])
    train_ = np.concatenate([
        train_[continuous_features].values,
        scaler.transform(train_[categorical_features]).toarray()
    ], axis=1)
    syndata_ = np.concatenate([
        syndata_[continuous_features].values,
        scaler.transform(syndata_[categorical_features]).toarray()
    ], axis=1)
    
    # Computing pair-wise distances between real and synthetic 
    dist_rf = metrics.pairwise_distances(train_, Y=syndata_, metric='minkowski', n_jobs=-1)
    
    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_indexes_rf = [dist_rf[i].argsort()[0] for i in range(len(dist_rf))]
    smallest_rf = [dist_rf[i][smallest_indexes_rf[i]] for i in range(len(dist_rf))]       
    
    return smallest_rf