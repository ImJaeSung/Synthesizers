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
Data Source: https://archive.ics.uci.edu/dataset/94/spambase
"""
#%%
class CustomDataset(Dataset): 
    def __init__(
        self,
        config,
        train=True):

        self.config = config
        self.train = train
        base = pd.read_csv('./data/spambase.data', header=None)
        columns = [
            "word_freq_make",
            "word_freq_address",
            "word_freq_all",
            "word_freq_3d",
            "word_freq_our",
            "word_freq_over",
            "word_freq_remove",
            "word_freq_internet",
            "word_freq_order",
            "word_freq_mail",
            "word_freq_receive",
            "word_freq_will",
            'word_freq_people',
            "word_freq_report",
            'word_freq_addresses',
            "word_freq_free",
            "word_freq_business",
            "word_freq_email",
            "word_freq_you",
            'word_freq_credit',
            'word_freq_your',
            "word_freq_font",
            'word_freq_000',
            'word_freq_money',
            "word_freq_hp",
            'word_freq_hpl',
            'word_freq_george',
            "word_freq_650",
            "word_freq_lab",
            "word_freq_labs",
            'word_freq_telnet',
            "word_freq_857",
            "word_freq_data",
            'word_freq_415',
            "word_freq_85",
            "word_freq_technology",
            "word_freq_1999",
            "word_freq_parts",
            'word_freq_pm',
            "word_freq_direct",
            "word_freq_cs",
            "word_freq_meeting",
            'word_freq_original',
            'word_freq_project',
            'word_freq_re',
            'word_freq_edu',
            'word_freq_table',
            'word_freq_conference',
            "char_freq_;",
            "char_freq_(",
            "char_freq_[",
            "char_freq_!",
            "char_freq_$",
            "char_freq_#",
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
            "class"
        ]
        base.columns = columns

        assert base.isna().sum().sum() == 0
    
        columns.remove("class")
        self.continuous_features = columns
        self.categorical_features = [
            "class"
        ]
        self.integer_features = [
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
        ]

        self.ClfTarget = "class"
        
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
