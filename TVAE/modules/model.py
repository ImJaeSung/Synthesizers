# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# %%
class TVAE(nn.Module):
    def __init__(self, config, device):
        super(TVAE, self).__init__()

        self.config = config
        self.device = device
        self.hidden_dim = 128

        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(config["input_dim"], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, config["latent_dim"] * 2),
        ).to(device)

        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dim"], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, config["input_dim"]),
        ).to(device)
        self.sigma = nn.Parameter(torch.ones(config["input_dim"]) * 0.1)

    def get_posterior(self, input):
        h = self.encoder(nn.Flatten()(input))
        mean, logvar = torch.split(h, self.config["latent_dim"], dim=1)
        return mean, logvar

    def encode(self, input):
        mean, logvar = self.get_posterior(input)
        noise = torch.randn(input.size(0), self.config["latent_dim"]).to(self.device)
        latent = mean + torch.exp(logvar / 2) * noise
        return mean, logvar, latent

    def forward(self, input):
        """encoding"""
        mean, logvar, latent = self.encode(input)
        """decoding"""
        xhat = self.decoder(latent)
        return mean, logvar, latent, xhat
    
    def generate_synthetic_data(self, n, train_dataset):
        steps = n // self.config["batch_size"] + 1
        data = []
        with torch.no_grad():
            for _ in range(steps):
                mean = torch.zeros(self.config["batch_size"], self.config["latent_dim"])
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self.device)
                fake = self.decoder(noise)
                fake = torch.tanh(fake)
                data.append(fake.cpu().numpy())
        data = np.concatenate(data, axis=0).astype(float)
        data = data[: len(train_dataset.raw_data)]
        
        syndata = train_dataset.EncodedInfo.transformer.inverse_transform(data, self.sigma.detach().cpu().numpy())
        
        # post-processing
        syndata[train_dataset.categorical_features] = syndata[train_dataset.categorical_features].astype(int)
        syndata[train_dataset.integer_features] = syndata[train_dataset.integer_features].round(0).astype(int)
        
        for feature in train_dataset.categorical_features:
            syndata[feature] = syndata[feature].apply(lambda x: 0 if x < 0 else x)

        return syndata

# %%
