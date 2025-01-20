#%%
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from collections import namedtuple

EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['dim', 'activation_fn']
)
#%%
"""
Data Source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
"""
#%%
class CustomDataset(Dataset): 
    def __init__(
        self,
        config,
        train=True):

        self.config = config
        self.train = train
        base = pd.read_csv('./data/Concrete_Data.csv')
        columns = [
            "Cement",
            "Blast Furnace Slag",
            "Fly Ash",
            "Water",
            "Superplasticizer",
            "Coarse Aggregate",
            "Fine Aggregate",
            "Age",
            "Concrete compressive strength"
        ]
        base.columns = columns

        assert base.isna().sum().sum() == 0
        
        columns.remove("Age")

        self.continuous_features = columns
        self.categorical_features = [
            "Age"
        ]
        self.integer_features = []

        self.ClfTarget = "Age"
        
        self.features = self.continuous_features + self.categorical_features
        base = base[self.features]

        # encoding categorical type
        base[self.categorical_features] = base[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        
        train_raw, test_raw = train_test_split(
            base, test_size=config["test_size"], random_state=config["seed"]
        )
        self.raw_data = train_raw if self.train else test_raw
        
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
        
        self.data, self.mean, self.std = self._Normalizer(
            train_data, 
            self.data
        ) 
        # Output Information
        self.EncodedInfo_list = []
        for _ in self.continuous_features:
            self.EncodedInfo_list.append(EncodedInfo(1, 'CRPS'))
        for _, dummy in zip(self.categorical_features, df_dummy):
            self.EncodedInfo_list.append(EncodedInfo(dummy.shape[1], 'softmax'))
        
    def _Normalizer(self, train_data, data):
        mean = train_data[self.continuous_features].mean(axis=0)
        std = train_data[self.continuous_features].std(axis=0)

        data[self.continuous_features] -= mean
        data[self.continuous_features] /= std

        data = data.astype(float)

        return data.values, mean, std
              
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        return torch.FloatTensor(self.data[idx])
#%%
