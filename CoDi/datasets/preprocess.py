"""Reference
[1] https://github.com/ChaejeongLee/CoDi/blob/main/tabular_transformer.py
"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from datasets.raw_data import load_raw_data

from collections import namedtuple
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['dim', 'activation_fn']
)
#%%
class CustomDataset(Dataset):
    def __init__(
        self, 
        config, 
        train=True):
        
        self.config = config
        self.train = train
        base, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config["dataset"])
        self.continuous_features = continuous_features
        self.num_continuous_features = len(continuous_features)
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget
        
        self.features = self.continuous_features + self.categorical_features
        base = base[self.features]

        # encoding for categorical variables.
        base[self.categorical_features] = base[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        self.num_categories = base[self.categorical_features].nunique(axis=0).to_list()
        
        # Data split
        train_raw, test_raw = train_test_split(
            base, test_size=config["test_size"], random_state=config["seed"])
        
        self.raw_data = train_raw[self.features] if self.train else test_raw[self.features]
        
        # one-hot encoding
        df_dummy = []
        for categorical_feature in self.categorical_features:
            df_dummy.append(pd.get_dummies(base[categorical_feature], prefix=categorical_feature, dtype=float))
        data = pd.concat([base.drop(columns=self.categorical_features)] + df_dummy, axis=1)
        
        # split encoding data
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"]
        )
        self.data = train_data if self.train else test_data
        
        # checking correct split
        assert sum(self.raw_data.index != self.data.index) == 0
        
        self.data, self.max, self.min = self._Normalizer(
            train_data, 
            self.data,
        ) 
        
        # Output Information
        self.EncodedInfo_list = []
        for _ in self.continuous_features:
            self.EncodedInfo_list.append(EncodedInfo(1, 'tanh'))
        for _, dummy in zip(self.categorical_features, df_dummy):
            self.EncodedInfo_list.append(EncodedInfo(dummy.shape[1], 'softmax'))
        
    def _Normalizer(self, train_data, data):
        max_ = train_data[self.continuous_features].max(axis=0)
        min_ = train_data[self.continuous_features].min(axis=0)
        range_ = max_ - min_
        
        data[self.continuous_features] -= min_
        data[self.continuous_features] /= range_
        
        # if self.config['activation_fn'] == 'tanh':
        data[self.continuous_features] = data[self.continuous_features] * 2 - 1
            # inverse transform: current = (current + 1) / 2
        data = data.astype(float)

        return data.values, max_, min_
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%