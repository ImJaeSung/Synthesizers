"""Reference:
[1] https://github.com/tennisonliu/GOGGLE/blob/main/src/goggle/model/Encoder.py
[2] https://github.com/tennisonliu/GOGGLE/blob/main/src/goggle/model/GraphDecoder.py
"""
#%%
import pandas as pd
import numpy as np
import torch
from torch import nn
# 3rd Party
from dgl.nn import GraphConv, SAGEConv

from modules.utils import (GraphInputProcessorHet, 
                           GraphInputProcessorHomo, 
                           LearnedGraph, 
                           RGCNConv)
#%%
class Encoder(nn.Module):
    def __init__(self, config):         
        super(Encoder, self).__init__()
        input_dim = config["input_dim"]
        encoder_dim = config["encoder_dim"]
        encoder_l = config["encoder_l"]
        
        encoder = nn.ModuleList(
            [nn.Linear(input_dim, encoder_dim), nn.ReLU()]
        )
        for _ in range(encoder_l - 2):
            encoder_dim_ = int(encoder_dim / 2)
            encoder.append(nn.Linear(encoder_dim, encoder_dim_))
            encoder.append(nn.ReLU())
            encoder_dim = encoder_dim_
        self.encoder = nn.Sequential(*encoder)
        self.encode_mu = nn.Linear(encoder_dim, input_dim)
        self.encode_logvar = nn.Linear(encoder_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu_z, logvar_z = self.encode_mu(h), self.encode_logvar(h)
        z = self.reparameterize(mu_z, logvar_z)
        return z, (mu_z, logvar_z)
    
#%%
class GraphDecoderHomo(nn.Module):
    def __init__(self, config):
        super(GraphDecoderHomo, self).__init__()
        decoder_dim = config["decoder_dim"]
        decoder_l = config["decoder_l"]
        
        decoder = nn.ModuleList([])
        if config["decoder_arch"] == "gcn":
            for i in range(decoder_l):
                if i == decoder_l - 1:
                    decoder.append(
                        GraphConv(
                            decoder_dim, 
                            1, 
                            norm="both", 
                            weight=True, 
                            bias=True
                        )
                    )
                else:
                    decoder_dim_ = int(decoder_dim / 2)
                    decoder.append(
                        GraphConv(
                            decoder_dim,
                            decoder_dim_,
                            norm="both",
                            weight=True,
                            bias=True,
                            activation=nn.Tanh(),
                        )
                    )
                    decoder_dim = decoder_dim_
        elif config["decoder_arch"] == "sage":
            for i in range(decoder_l):
                if i == decoder_l - 1:
                    decoder.append(
                        SAGEConv(
                            decoder_dim, 
                            1, 
                            aggregator_type="mean", 
                            bias=True
                        )
                    )
                else:
                    decoder_dim_ = int(decoder_dim / 2)
                    decoder.append(
                        SAGEConv(
                            decoder_dim,
                            decoder_dim_,
                            aggregator_type="mean",
                            bias=True,
                            activation=nn.Tanh(),
                        )
                    )
                    decoder_dim = decoder_dim_
        else:
            raise Exception("decoder can only be {het|gcn|sage}")

        self.decoder = nn.Sequential(*decoder)

    def forward(self, graph_input, b_size):
        b_z, b_adj, b_edge_weight = graph_input

        for layer in self.decoder:
            b_z = layer(b_adj, feat=b_z, edge_weight=b_edge_weight)

        x_hat = b_z.reshape(b_size, -1)

        return x_hat
#%%
class GraphDecoderHet(nn.Module):
    def __init__(self, config, n_edge_types):
        super(GraphDecoderHet, self).__init__()
        decoder_dim = config["decoder_dim"]
        decoder_l = config["decoder_l"]
 
        decoder = nn.ModuleList([])
        for i in range(decoder_l):
            if i == decoder_l - 1:
                decoder.append(
                    RGCNConv(
                        decoder_dim,
                        1,
                        num_relations=n_edge_types + 1,
                        root_weight=False,
                    )
                )
            else:
                decoder_dim_ = int(decoder_dim / 2)
                decoder.append(
                    RGCNConv(
                        decoder_dim,
                        decoder_dim_,
                        num_relations=n_edge_types + 1,
                        root_weight=False,
                    )
                )
                decoder.append(nn.ReLU())
                decoder_dim = decoder_dim_

        self.decoder = nn.Sequential(*decoder)

    def forward(self, graph_input, b_size):
        b_z, b_edge_index, b_edge_weights, b_edge_types = graph_input

        h = b_z
        for layer in self.decoder:
            if not isinstance(layer, nn.ReLU):
                h = layer(h, b_edge_index, b_edge_types, b_edge_weights)
            else:
                h = layer(h)

        x_hat = h.reshape(b_size, -1)

        return x_hat

#%%
class Goggle(nn.Module):
    def __init__(
        self,
        config,
        device):
        super(Goggle, self).__init__()
        self.input_dim = config["input_dim"]
        self.config = config 
        self.device = device
        self.learned_graph = LearnedGraph(config, device)
        self.encoder = Encoder(config)
        
        if config["decoder_arch"] == "het":
            n_edge_types = config["input_dim"] * config["input_dim"]
            self.graph_processor = GraphInputProcessorHet(
                config, n_edge_types, device
            )
            self.decoder = GraphDecoderHet(
                config, n_edge_types
            )
        else:
            self.graph_processor = GraphInputProcessorHomo(
                config, device
            )
            self.decoder = GraphDecoderHomo(
                config
            )

    def forward(self, x, iter):
        z, (mu_z, logvar_z) = self.encoder(x)
        b_size, _ = z.shape
        adj = self.learned_graph(iter)
        graph_input = self.graph_processor(z, adj)
        x_hat = self.decoder(graph_input, b_size)

        return x_hat, adj, mu_z, logvar_z

    def generate_synthetic_data(self, n, train_dataset):
        data = []
        steps = n // self.config['batch_size']
        
        with torch.no_grad():
            for _ in range(steps):
                mu = torch.zeros(self.input_dim)
                sigma = torch.ones(self.input_dim)
                q = torch.distributions.Normal(mu, sigma)
                z = q.rsample(
                    sample_shape=torch.Size([self.config['batch_size']])
                ).squeeze().to(self.device)

                self.learned_graph.eval()
                self.graph_processor.eval()
                self.decoder.eval()

                adj = self.learned_graph(100)
                graph_input = self.graph_processor(z, adj)
                samples = self.decoder(graph_input, self.config['batch_size'])
            data.append(samples)
        data = torch.cat(data, dim=0)
        data = data[:n, :]
        
        data = pd.DataFrame(data.cpu().numpy(), columns=train_dataset.continuous_features + train_dataset.categorical_features)
        
        """un-standardization of synthetic data"""
        for col, scaler in train_dataset.cont_scalers.items():
            data[[col]] = scaler.inverse_transform(data[[col]])
        
        """post-process integer columns (calibration)"""
        data[train_dataset.integer_features] = data[train_dataset.integer_features].round(0).astype(int)
        data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
        
        return data
# %%
