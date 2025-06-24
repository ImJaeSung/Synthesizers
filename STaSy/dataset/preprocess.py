#%%
import pandas as pd
import numpy as np
from collections import namedtuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from dataset.raw_data import load_raw_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler,
                                   QuantileTransformer,
                                   OrdinalEncoder,
                                   OneHotEncoder, 
                                   MinMaxScaler)
from sklearn.pipeline import make_pipeline

EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'num_categories'])

#%% for TabSyn part
def task_type(config):
    if config["dataset"] in ["banknote", "breast", "bankruptcy", "default"]:
        task_type = "binclass"
    else:
        task_type = "regression"
    return task_type


class CustomDataset(Dataset):
    def __init__(
        self, 
        config, 
        cont_scalers=None,
        cat_scalers=None,
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
        
        self.num_col_idx = [self.col_2_idx[col] for col in self.continuous_features]
        self.cat_col_idx = [self.col_2_idx[col] for col in self.categorical_features ]#if col != self.ClfTarget
        self.target_col_idx = [self.col_2_idx[col] for col in [self.ClfTarget]]
        
        # encoding categorical dataset
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        data = data[self.features] # select features for training
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"])
        
        data = train_data if train else test_data
        data = data.reset_index(drop=True)
        self.raw_data = train_data if train else test_data
        
        ### scaling part
        self.cont_scalers = {} if train else cont_scalers
        cont_transformed = []
        for continuous_feature in tqdm(self.continuous_features, desc="Tranform Continuous Features..."):
            cont_transformed.append(self.transform_continuous(data, continuous_feature))
        
        self.cat_scalers = {} if train else cat_scalers
        cat_transformed = []
        for categorical_feature in tqdm(self.categorical_features, desc="Tranform Categorical Features..."):
            cat_transformed.append(self.transform_categorical(data, categorical_feature))
        
        self.data = np.concatenate(
            cont_transformed + cat_transformed, axis=1
        )
        
        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, self.num_categories)
        
        self.category_maps = {}
        self.X_num = self.data[:,self.num_col_idx].astype(np.float32)
        self.X_cat = np.concatenate(cat_transformed, axis=1).astype(int)
        self.y = self.data[:,self.target_col_idx]
        
        self.idx_mapping, self.inverse_idx_mapping, self.idx_name_mapping  = get_column_name_mapping(
            self.raw_data,
            self.num_col_idx,
            self.cat_col_idx,
            self.target_col_idx,
            column_names = None)

        self.info = {}
        self.info["task_type"] = "binclass"
        self.info["dataset"] = config['dataset']
        self.info['num_col_idx'] = self.num_col_idx
        self.info['cat_col_idx'] = self.cat_col_idx
        self.info['target_col_idx'] = self.target_col_idx
        self.info['idx_mapping'] = self.idx_mapping
        self.info['inverse_idx_mapping'] = self.inverse_idx_mapping
        self.info['idx_name_mapping'] = self.idx_name_mapping
        config['info'] = self.info
        
    def transform_continuous(self, data, col):
        feature = data[col].values.reshape(-1, 1)   

        if self.train:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(feature)
            self.cont_scalers[col] = scaler
        else:
            scaler = self.cont_scalers[col]

        transformed_feature = scaler.transform(feature)
        return transformed_feature
    
    def transform_categorical(self, data, col):
        feature = data[[col]]
        self.unknown_value = np.iinfo(np.int64).max - 3  
        if self.train:
            ohe = OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, dtype=np.float32 # type: ignore[code]
            )

            encoder = make_pipeline(ohe)
            encoder.fit(feature)  
            self.cat_scalers[col] = encoder  

            transformed_feature = encoder.transform(feature).astype(np.int64)

        else:  
            encoder = self.cat_scalers[col]  
            transformed_feature = encoder.transform(feature).astype(np.int64)

        return transformed_feature
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_num = self.X_num[idx]
        x_cat = self.X_cat[idx]

        x_num = torch.tensor(x_num, dtype=torch.float32)
        x_cat = torch.tensor(x_cat, dtype=torch.long)
        return x_num, x_cat    

def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping 
#%%
    
   
def build_num_inverse_fn(scalers):
    def num_inverse_fn(X_num: np.ndarray) -> np.ndarray:
        restored_cols = []
        for i, (col_name, scaler) in enumerate(scalers.items()):
            col_data = X_num[:, i].reshape(-1, 1)  # (N,1) 형태로 변환
            col_restored = scaler.inverse_transform(col_data)
            restored_cols.append(col_restored)
        return np.hstack(restored_cols)
    
    return num_inverse_fn


def build_cat_inverse_fn(scalers):
    def cat_inverse_fn(X_cat: np.ndarray) -> np.ndarray:
        restored_cols = []
        
        for i, (col_name, scaler) in enumerate(scalers.items()):
            col_data = X_cat[:, i].reshape(-1, 1)   
            categories = scaler.named_steps["ordinalencoder"].categories_[0]   

            max_index = len(categories) - 1   
            col_data = np.clip(col_data, 0, max_index)  
            
            col_restored = scaler.inverse_transform(col_data)

            restored_cols.append(col_restored)

        return np.hstack(restored_cols)
    
    return cat_inverse_fn