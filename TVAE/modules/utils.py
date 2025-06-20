#%%
import torch
import numpy as np
import random

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
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
    
def memorization_ratio(train, syndata, continuous_features, categorical_features):
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

    dist = metrics.pairwise_distances(syndata_, Y=train_, n_jobs=-1)
    dist = np.sort(dist, axis=1)
    EPS = 1e-8
    ratio = dist[:, 0] / (dist[:, 1] + EPS)
    
    return ratio 