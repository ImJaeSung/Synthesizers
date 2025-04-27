#%%
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from modules.data_transformer import DataTransformer
from datasets.raw_data import load_raw_data

from collections import namedtuple
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['transformer', 'continuous_features', 'categorical_features'])
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
        data = data[self.features]

        # encoding for categorical variables.
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        
        # Data split
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"])
        
        data = train_data if self.train else test_data
        data = data.reset_index(drop=True)
        
        raw_data = train_data[self.features] if self.train else test_data[self.features]
        self.raw_data = raw_data.reset_index(drop=True)

        # transformation
        transformer = DataTransformer()
        transformer.fit(train_data, discrete_columns=self.categorical_features, random_state=config["seed"])
        self.data = transformer.transform(train_data)
        
        # save information
        self.EncodedInfo = EncodedInfo(
            transformer, self.continuous_features, self.categorical_features)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%