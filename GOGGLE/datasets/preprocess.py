#%%
import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

import torch
from torch.utils.data import (Dataset, DataLoader, TensorDataset)

# from modules.data_transformer import DataTransformer
from datasets.raw_data import load_raw_data

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import namedtuple
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['continuous_features', 'categorical_features', 'num_features'])
#%%
class CustomDataset(Dataset):
    def __init__(
        self, 
        config, 
        cont_scalers=None,
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
        self.num_categories = data[categorical_features].nunique(axis=0).to_list()
        
        # Data split
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"])
        
        data = train_data if self.train else test_data
        data = data.reset_index(drop=True)

        raw_data = train_data[self.features] if self.train else test_data[self.features]
        self.raw_data = raw_data.reset_index(drop=True)

        # Continuous features: Standard Scaler
        self.cont_scalers = {} if train else cont_scalers
        cont_transformed = []
        for continuous_feature in tqdm(continuous_features, desc="Tranform Continuous Features..."):
            cont_transformed.append(
                self.transform_continuous(data, continuous_feature)
            )
        #%%
        self.data = np.concatenate(
            (np.vstack(cont_transformed).T, np.array(data.iloc[:,len(self.continuous_features):])), axis=1
        )
        # # transformation
        # if scalers:
        #     ind = list(range(len(data.columns)))
        #     ind = [x for x in ind if x != data.columns.get_loc(self.categorical_features[0])]
        #     col_list = data.columns[ind]
        #     ct = ColumnTransformer(
        #         [("scaler", StandardScaler(), col_list)], remainder="passthrough"
        #     )

        #     X_ = ct.fit_transform(data)
        #     self.data = pd.DataFrame(X_, index=data.index, columns=data.columns)
        # save information
        self.EncodedInfo = EncodedInfo(
            self.continuous_features, self.categorical_features, len(self.features)
        )
    #%%
    def transform_continuous(self, data, col):
        feature = data[[col]].to_numpy().astype(float)
        
        if self.train:
            cont_scaler = StandardScaler().fit(feature)
            self.cont_scalers[col] = cont_scaler
        else:
            cont_scaler = self.cont_scalers[col]
            
        transformed = cont_scaler.transform(feature)[:, 0]
        
        return transformed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%
# data = pd.read_csv('/root/default/SyntheticTabular/data/breast.csv')
# data = data.drop(columns=['id']) # drop ID number
# assert data.isna().sum().sum() == 0

# continuous_features = [x for x in data.columns if x != "diagnosis"]
# categorical_features = ["diagnosis"]
# integer_features = []
# ClfTarget = "diagnosis"

# CustomDataset(config)

# data[categorical_features] = data[categorical_features].apply(
#             lambda col: col.astype('category').cat.codes)

# X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)

# train_dataset = TensorDataset(torch.Tensor(X_train.values))
# X_dataset = TensorDataset(torch.Tensor(X_test.values))

# train_dataloader = get_dataloader(train_dataset,32,0)

# from synthcity.plugins.core.schema import Schema

# schema = Schema(data=data)
# schema.as_constraints()
# data['diagnosis']
# for i in data.columns:
#     if i=='diagnosis':
#         print(data[i])
#         break

# data['diagnosis'].unique().tolist()
#%%
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(X, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)

    train_dataloader = DataLoader(
        X,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_dataloader    