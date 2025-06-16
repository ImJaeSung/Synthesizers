#%%
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

#%%
class OrderedEmbedding(nn.Module) :
    def __init__(
        self, 
        occ : torch.Tensor, 
        dim_transformer : int):
        
        """ 
        occ : ordered cluster centers k 
        """
        super().__init__()
        self.E = nn.Parameter(torch.zeros(len(occ), dim_transformer)) # unordered embedding matrix : kxd
        self.l = nn.Parameter(torch.randn(dim_transformer) * 0.05) # endpoint vector : d
        self.h = nn.Parameter(torch.randn(dim_transformer) * 0.05) # endpoint vector : d
        self.r = (occ - occ[0]) / (occ[-1] - occ[0]) # min-max scaling : k

    @property
    def weight(self):
        return torch.stack([self.r[idx] * self.l + (1 - self.r[idx] ) * self.h + self.E[idx] 
                            for idx in range(len(self.r))])
    def forward (self, idx):
        return self.weight[idx]


class Embedding(nn.Module):
    def __init__(self, config, EncodedInfo, device):
        super().__init__()
        self.config = config
        self.EncodedInfo = EncodedInfo
        self.device = device

        # self.OrderedEmbed = nn.ModuleList(
        #     [OrderedEmbedding(occ, config["dim_transformer"]) for occ in EncodedInfo.OCC]
        #     )
        self.OrderedEmbed = nn.ModuleList()
        for occ in EncodedInfo.OCC:
            self.OrderedEmbed.append(
                OrderedEmbedding(occ, config["dim_transformer"]).to(device)
            )

        self.DiscEmbed = nn.ModuleList()
        for num_category in EncodedInfo.num_categories:
            self.DiscEmbed.append(
                nn.Embedding(num_category, config["dim_transformer"]).to(device)
            )

        self.MaskEmbed = nn.Parameter(torch.randn(config["dim_transformer"])*0.05).to(device)

        self.PosEmbed = nn.Embedding(
            num_embeddings=EncodedInfo.num_continuous_features + len(EncodedInfo.num_categories), 
            embedding_dim=config["dim_transformer"]).to(device)
        
        self.init_weights()

    def init_weights(self):
        for layer in self.DiscEmbed:
            nn.init.normal_(layer.weight, mean = 0.0, std = 0.05) 

        nn.init.normal_(self.PosEmbed.weight, mean = 0.0, std = 0.01)


    def forward(self, batch, mask):
        # continuous
        continuous = batch[:, :self.EncodedInfo.num_continuous_features]
        continuous_embedded = torch.stack(
            [self.OrderedEmbed[i](continuous[:, i]) for i in range(len(continuous[0]))]
        ).transpose(0, 1)

        # categorical
        categorical = batch[:, self.EncodedInfo.num_continuous_features:]
        categorical_embedded = torch.stack(
            [self.DiscEmbed[i](categorical[:, i]) for i in range(len(categorical[0]))]
        ).transpose(0, 1)
        
        embedded = torch.cat([continuous_embedded, categorical_embedded], dim=1)
        embedded[mask] = self.MaskEmbed
        embedded += self.PosEmbed.weight 
        return embedded
    
class DynamicLinear(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.E = embedding
        self.bias = nn.Parameter(torch.zeros(len(embedding.weight)))
        self.temp = nn.Parameter(torch.ones(1))

    def forward (self , x):
        h = x@self.E.weight.T + self.bias
        return h / torch.sigmoid(self.temp)

class DynamicLinearLayer(nn.Module):
    def __init__(self, config, EncodedInfo, device):
        super().__init__()

        self.embedding = nn.ModuleList()
        for occ in EncodedInfo.OCC:
            self.embedding.append(
                OrderedEmbedding(occ, config["dim_transformer"]).to(device)
            ) # [occ, dim]
        for num_category in EncodedInfo.num_categories:
            self.embedding.append(
                nn.Embedding(num_category, config["dim_transformer"]).to(device)
            )

        self.dynamic_linear = nn.ModuleList()
        for embedding in self.embedding:
            self.dynamic_linear.append(DynamicLinear(embedding).to(device))

    def forward(self, x):
        return [
            self.dynamic_linear[i](x[:, i, :]) for i in range(len(self.dynamic_linear))
        ] 
    
class TabMT(nn.Module):
    def __init__(self, config, EncodedInfo, device):
        super().__init__()
        self.config = config
        self.EncodedInfo = EncodedInfo
        self.device = device

        self.embedding = Embedding(config, EncodedInfo, device)
        
        self.transformer = nn.ModuleList()
        for _ in range(config["num_transformer_layer"]):
            self.transformer.append(
                nn.TransformerEncoderLayer(
                    d_model = config["dim_transformer"], 
                    nhead = config["num_transformer_heads"], 
                    dropout = config["transformer_dropout"], 
                    batch_first = True).to(device)
            )
        self.dynamic_linear = DynamicLinearLayer(config, EncodedInfo, device)

    def forward(self, batch, mask):
        x = self.embedding(batch, mask)
        for layer in self.transformer:
            x = layer(x)
        pred = self.dynamic_linear(x)
        return pred
    
    def generate_synthetic_data(self, n, train_dataset):
        data = []
        steps = n // self.config["batch_size"] + 1
        
        for _ in tqdm(range(steps), desc="Generate Synthetic Dataset..."):
            with torch.no_grad():
                batch = torch.zeros(
                    self.config["batch_size"], train_dataset.EncodedInfo.num_features
                ).long().to(self.device)

                syn_data = torch.zeros(
                self.config["batch_size"], train_dataset.EncodedInfo.num_features
                ).float().to(self.device)

                mask = torch.ones_like(batch).bool().to(self.device)

                for num_feature in torch.randperm(train_dataset.EncodedInfo.num_features):
                    preds = self(batch, mask)
                    if num_feature < len(train_dataset.EncodedInfo.OCC):
                        cat_label = Categorical(logits=preds[num_feature] / self.config["tau"]).sample()
                        batch[: , num_feature] = cat_label
                        syn_data[: , num_feature] = train_dataset.EncodedInfo.OCC[num_feature].to(self.device)[cat_label]
                    else:
                        batch[: , num_feature] = Categorical(logits=preds[num_feature] / self.config["tau"]).sample()
                        syn_data[: , num_feature] = batch[: , num_feature]
                    batch [: , num_feature] = Categorical(logits=preds[num_feature] / self.config["tau"]).sample()
                    mask [: , num_feature] = False
            
            data.append(syn_data)

        data = torch.cat(data, dim=0)
        data = data[:n, :]
        data = pd.DataFrame(data.cpu().numpy(), columns=train_dataset.features)
        
        data[train_dataset.continuous_features] = data[train_dataset.continuous_features].astype(float) # align the synthetic-eval package
        data[train_dataset.integer_features] = data[train_dataset.integer_features].astype(int)
        data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
        
        return data
