#%%
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import namedtuple

from datasets.raw_data import load_raw_data

#%%
EncodedInfo = namedtuple(
    'EncodedInfo',
    ['type', 'continuous_features', 'categorical_features', 'integer_features']
)
#%%
#%%
class CustomDataset(Dataset):
    def __init__(
        self, 
        config, 
        train=True):
        
        self.config = config
        self.train = train
        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config["dataset"])
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget
        
        self.features = self.continuous_features + self.categorical_features
        self.col_2_idx = {col : i for i, col in enumerate(data[self.features].columns.to_list())}
        self.num_continuous_features = len(self.continuous_features)
        
        # 범주형 데이터 인코딩
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        # 필요한 컬럼만 정렬 및 훈련 테스트 분할
        data = data[self.features] # select features for training
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"])
        
        data = train_data if train else test_data
        self.data = data.reset_index(drop=True)
        self.raw_data = self.data
        
        type = {"Classification": self.ClfTarget}
        # save information
        self.EncodedInfo = EncodedInfo(
            type, continuous_features, categorical_features, integer_features)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%
