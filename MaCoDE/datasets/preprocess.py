#%%
import numpy as np
from collections import namedtuple
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

import torch
from torch.utils.data import Dataset

from modules.missing import generate_mask
from datasets.raw_data import load_raw_data

EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'num_categories'])
#%%
class CustomDataset(Dataset):
    def __init__(
        self, 
        config, 
        scalers=None,
        train=True):
        
        self.config = config
        self.train = train
        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config)
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget
        
        self.features = self.continuous_features + self.categorical_features
        self.col_2_idx = {col : i for i, col in enumerate(data[self.features].columns.to_list())}
        self.num_continuous_features = len(self.continuous_features)
        
        # categorical column encoding
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes + 1) # "0" for [MASK] token
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        data = data[self.features] # select features for training
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"])
        
        data = train_data if train else test_data
        data = data.reset_index(drop=True)
        self.raw_data = train_data[self.features] if train else test_data[self.features]
        
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
        
        self.scalers = {} if train else scalers
        self.bins = np.linspace(0, 1, self.config["bins"]+1, endpoint=True)
        print(f"The number of bins: {len(self.bins)-1}")
        transformed = []
        for continuous_feature in tqdm(self.continuous_features, desc="Tranform Continuous Features..."):
            transformed.append(self.transform_continuous(data, continuous_feature))
        
        self.data = np.concatenate(
            transformed + [data[self.categorical_features].values], axis=1
        )
        
        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, self.num_categories)
    
    def transform_continuous(self, data, col):
        nan_value = data[[col]].to_numpy().astype(float)
        nan_mask = np.isnan(nan_value)
        feature = nan_value[~nan_mask].reshape(-1, 1)
        
        if self.train:
            scaler = QuantileTransformer(
                output_distribution='uniform',
                subsample=None,
            ).fit(feature)
            self.scalers[col] = scaler
        else:
            scaler = self.scalers[col]
        
        nan_value[nan_mask] = 0. # replace NaN with arbitrary value
        transformed = scaler.transform(nan_value)
        transformed = np.where(
            transformed == 1, 1-1e-6, transformed
        ) # maximum value will be assinged to the last bin
        transformed = np.digitize(
            transformed, self.bins
        ).astype(float) # range = (1, 2, ..., #bins) ("0" for [MASK] token) 
        transformed[nan_mask] = np.nan
        return transformed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%
