
#%%
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import torch
from torch.utils.data import Dataset

from modules.missing import generate_mask
from datasets.raw_data import load_raw_data
#%%
from collections import namedtuple
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'OCC', 'num_categories'])
#%%
class CustomDataset(Dataset):
    def __init__(
        self,
        config,
        kmeans_models=None,
        train=True):

        self.config = config
        self.train = train
        
        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config)
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.features = self.continuous_features + self.categorical_features
        self.ClfTarget = ClfTarget
        
        self.col_2_idx = {col : i for i, col in enumerate(data[self.features].columns.to_list())}
        self.num_continuous_features = len(self.continuous_features)

        # encoding for categorical type
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        # split train and test
        data = data[self.features] # select features for training
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"])
        
        data = train_data if self.train else test_data
        data = data.reset_index(drop=True)
        self.raw_data = train_data[self.features] if self.train else test_data[self.features] # save the raw data
 
        # generate missingness patterns
        if train:
            if config["missing_type"] != "None":
                mask = generate_mask(
                    torch.from_numpy(data.values).float(), 
                    config["missing_rate"], 
                    config["missing_type"],
                    seed=config["seed"])
                data.mask(mask.astype(bool), np.nan, inplace=True)
                self.mask = mask 

        self.kmeans_models = {} if self.train else kmeans_models
        self.data = self.transform_continuous(data)

    def transform_continuous(self, data):
        # quantizing and encoding for continuous
        OCC = []
        continuous_data = data[self.continuous_features]

        for continuous_feature in tqdm(self.continuous_features, desc="Transform Continuous Features..."):
            nan_value = continuous_data[[continuous_feature]].to_numpy()
            nan_mask = np.isnan(nan_value)
            feature = nan_value[~nan_mask].reshape(-1, 1)
            
            if self.train:
                kmeans = KMeans(
                    n_clusters=self.config["max_clusters"], random_state=self.config["seed"], n_init = 'auto')
                self.kmeans_models[continuous_feature] = kmeans.fit(feature) # save kmeans models fitted train
                kmeans = self.kmeans_models[continuous_feature]
            else:
                kmeans = self.kmeans_models[continuous_feature]
            
            centroids = kmeans.cluster_centers_
                        
            # replacing 0 for Nans
            nan_value[nan_mask] = 0.
            labels = kmeans.predict(nan_value)

            # ordering cluster centers
            sorted_indices = np.argsort(centroids.reshape(-1))
            sorted_centroids = centroids[sorted_indices]
            sorted_labels = np.zeros_like(labels)

            for i, sorted_idx in enumerate(sorted_indices):
                sorted_labels[labels == sorted_idx] = i

            # replacing -1 for NaNs
            sorted_labels[nan_mask.squeeze()] = -1
            continuous_data.loc[:,continuous_feature] = sorted_labels
            OCC.append(torch.tensor(sorted_centroids.squeeze(), dtype=torch.float32)) 
        
        data = pd.concat([continuous_data, data[self.categorical_features].fillna(-1)], axis=1).values
        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, OCC, self.num_categories)
        
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx])