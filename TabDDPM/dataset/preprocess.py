#%%
from tqdm import tqdm
from collections import namedtuple
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from dataset.raw_data import load_raw_data
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
#%%
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'num_categories'])
#%%
class CustomDataset(Dataset):
    def __init__(
            self, 
            config, 
            train=True, 
            cont_scalers=None, 
            disc_scalers=None
        ):

        self.config = config
        self.train = train

        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config)
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget
        
        self.features = continuous_features + categorical_features
        self.num_continuous_features = len(self.continuous_features)
        
        # encoding for categorical variables.
        data[categorical_features] = data[categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        categorical_features.remove(self.ClfTarget)
        
        self.num_categories = data[categorical_features].nunique(axis=0).to_list()
        self.num_classes = data[self.ClfTarget].nunique()

        self.features.remove(self.ClfTarget)
        x_data = data[self.features]
        y_data = data[self.ClfTarget]

        x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(
            x_data, y_data, test_size=config["test_size"], random_state=config["seed"]
        )

        x_data = x_train_data if train else x_test_data
        y_data = y_train_data if train else y_test_data

        self.x_raw_data = x_train_data[self.features] if train else x_test_data[self.features]
        self.y_raw_data = y_train_data if train else y_test_data

        x_data = x_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)
        #%%
        # Continuous features: Quantile Transformation
        self.cont_scalers = {} if train else cont_scalers
        cont_transformed = []
        for continuous_feature in tqdm(continuous_features, desc="Tranform Continuous Features..."):
            cont_transformed.append(
                self.transform_continuous(x_data, continuous_feature, config)
            )

        #%%
        # Categorical feature: One-Hot encoding
        self.disc_scalers = {} if train else disc_scalers
        disc_transformed = []

        for categorical_feature in tqdm(categorical_features, desc="Tranform Categorical Features..."):
            disc_transformed.append(
                self.transform_categorical(x_data, categorical_feature, config)
            )
        #%%
        self.data = np.concatenate(
            (np.vstack(cont_transformed).T, np.hstack(disc_transformed)), axis=1
        )
        self.y = y_data

        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, self.num_categories)
    #%%
    def transform_continuous(self, data, col, config):
        feature = data[[col]].to_numpy().astype(float)

        if self.train:
            scaler = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(data.shape[0] // 30, 1000), 10),
                # subsample=1e9,
                random_state=config["seed"]
            )
            scaler.fit(feature)
            self.cont_scalers[col] = scaler
        else:
            scaler = self.scalers[col]

        transformed = scaler.transform(feature)[:, 0]
        
        return transformed
    #%%
    def transform_categorical(self, data, col, config):
        feature = data[[col]].to_numpy().astype(float)
        if self.train:
            scaler = OneHotEncoder(
                handle_unknown='ignore', 
                # sparse=False, 
                dtype=np.float32 # type: ignore[code]
            )
            scaler.fit(feature)
            self.disc_scalers[col] = scaler
        else:
            scaler = self.scalers[col]

        transformed = scaler.transform(feature).toarray()
        return transformed
    #%%
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.FloatTensor([self.y[idx]])  # Ensure label is a sequence
        return data, label
# %%