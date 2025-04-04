"""
Reference:
[1] https://github.com/an-seunghwan/DistVAE/blob/main/modules/model.py
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
#%%
class VAE(nn.Module):
    def __init__(self, config, device):
        super(VAE, self).__init__()
        
        self.config = config
        self.device = device

        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(config["CRPS_dim"] + config["softmax_dim"], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, config["latent_dim"] * 2),
        ).to(device)
        
        """spline"""
        self.delta = torch.arange(0, 1 + config["step"], step=config["step"]).view(1, -1).to(device)
        self.M = self.delta.size(1) - 1
        self.spline = nn.Sequential(
            nn.Linear(config["latent_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, config["CRPS_dim"] * (1 + (self.M + 1)) + config["softmax_dim"]),
        ).to(device)
    
    def get_posterior(self, input):
        h = self.encoder(input)
        mean, logvar = torch.split(h, self.config["latent_dim"], dim=1)
        return mean, logvar
    
    def sampling(self, mean, logvar, deterministic=False):
        if deterministic:
            z = mean
        else:
            noise = torch.randn(mean.size(0), self.config["latent_dim"]).to(self.device) 
            z = mean + torch.exp(logvar / 2) * noise
        return z
    
    def encode(self, input, deterministic=False):
        mean, logvar = self.get_posterior(input)
        z = self.sampling(mean, logvar, deterministic=deterministic)
        return z, mean, logvar
    
    def quantile_parameter(self, z):
        h = self.spline(z)
        logit = h[:, -self.config["softmax_dim"]:]
        spline = h[:, :-self.config["softmax_dim"]]
        h = torch.split(spline, 1 + (self.M + 1), dim=1)
        
        gamma = [h_[:, [0]] for h_ in h]
        beta = [nn.Softplus()(h_[:, 1:]) for h_ in h] # positive constraint
        return gamma, beta, logit
    
    def quantile_function(self, alpha, gamma, beta, j):
        return gamma[j] + (beta[j] * torch.where(alpha - self.delta > 0,
                                                alpha - self.delta,
                                                torch.zeros(()).to(self.device))).sum(axis=1, keepdims=True)
        
    def _quantile_inverse(self, x, gamma, beta, j):
        delta_ = self.delta.unsqueeze(2).repeat(1, 1, self.M + 1)
        delta_ = torch.where(delta_ - self.delta > 0,
                            delta_ - self.delta,
                            torch.zeros(()).to(self.device))
        mask = gamma[j] + (beta[j] * delta_.unsqueeze(2)).sum(axis=-1).squeeze(0).t()
        mask = torch.where(mask <= x, 
                        mask, 
                        torch.zeros(()).to(self.device)).type(torch.bool).type(torch.float)
        alpha_tilde = x - gamma[j]
        alpha_tilde += (mask * beta[j] * self.delta).sum(axis=1, keepdims=True)
        alpha_tilde /= (mask * beta[j]).sum(axis=1, keepdims=True) + 1e-6
        alpha_tilde = torch.clip(alpha_tilde, self.config["threshold"], 1) # numerical stability
        return alpha_tilde

    def quantile_inverse(self, x, gamma, beta):
        alpha_tilde_list = []
        for j in range(self.config["CRPS_dim"]):
            alpha_tilde = self._quantile_inverse(x[:, [j]], gamma, beta, j)
            alpha_tilde_list.append(alpha_tilde)
        return alpha_tilde_list
    
    def forward(self, input, deterministic=False):
        z, mean, logvar = self.encode(input, deterministic=deterministic)
        gamma, beta, logit = self.quantile_parameter(z)
        return z, mean, logvar, gamma, beta, logit
    
    def gumbel_sampling(self, size, eps = 1e-20):
        U = torch.rand(size)
        G = (- (U + eps).log() + eps).log()
        return G
    
    def generate_synthetic_data(self, n, EncodedInfo_list, train_dataset, reverse_col=False):
        data = []
        steps = n // self.config["batch_size"] + 1
        
        with torch.no_grad():
            for _ in range(steps):
                randn = torch.randn(self.config["batch_size"], self.config["latent_dim"]).to(self.device) # prior
                gamma, beta, logit = self.quantile_parameter(randn)
                
                samples = []
                st = 0
                for j, info in enumerate(EncodedInfo_list):
                    if info.activation_fn == "CRPS":
                        alpha = torch.rand(self.config["batch_size"], 1).to(self.device)
                        samples.append(self.quantile_function(alpha, gamma, beta, j))
                        
                    elif info.activation_fn == "softmax":
                        ed = st + info.dim
                        out = logit[:, st : ed]
                        
                        """Gumbel-Max Trick"""
                        G = self.gumbel_sampling(out.shape)
                        G = G.to(self.device)
                        _, out = (nn.LogSoftmax(dim=1)(out) + G).max(dim=1)
                        
                        samples.append(out.unsqueeze(1))
                        # samples.append(F.one_hot(out, num_classes=info.dim))
                        st = ed
            
                samples = torch.cat(samples, dim=1)
                data.append(samples)
        data = torch.cat(data, dim=0)
        data = data[:n, :]
        data = pd.DataFrame(data.cpu().numpy(), columns=train_dataset.continuous_features + train_dataset.categorical_features)
        
        """un-standardization of synthetic data"""
        data[train_dataset.continuous_features] = data[train_dataset.continuous_features] * train_dataset.std + train_dataset.mean
        
        """post-process integer columns (calibration)"""
        data[train_dataset.integer_features] = data[train_dataset.integer_features].round(0).astype(int)
        data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
        
        # if reverse_col:
        #     """reverse to original column names"""
        #     for dis, disdict in zip(train_dataset.categorical_features, train_dataset.discrete_dicts_reverse):
        #         data[dis] = data[dis].apply(lambda x:disdict.get(x))
        
        return data
#%%
def main():
    #%%
    config = {
        # "input_dim": 10,
        "latent_dim": 2,
        "step": 0.1,
        "CRPS_dim": 10,
        "softmax_dim": 7,
    }
    
    model = VAE(config, 'cpu')
    for x in model.parameters():
        print(x.shape)
    batch = torch.rand(10, config["CRPS_dim"] + config["softmax_dim"])
    
    z, mean, logvar, gamma, beta, logit = model(batch)
    
    j = 0
    delta_ = model.delta.unsqueeze(2).repeat(1, 1, model.M + 1)
    delta_ = torch.where(delta_ - model.delta > 0,
                        delta_ - model.delta,
                        torch.zeros(()))
    mask1 = gamma[j] + (beta[j] * delta_.unsqueeze(2)).sum(axis=-1).squeeze(0).t()
    
    mask2 = [model.quantile_function(d, gamma, beta, j) for d in model.delta[0]]
    mask2 = torch.cat(mask2, axis=1)
    
    assert (mask1 - mask2).sum().item() == 0
    
    assert z.shape == (10, config["latent_dim"])
    assert mean.shape == (10, config["latent_dim"])
    assert logvar.shape == (10, config["latent_dim"])
    assert gamma[0].shape == (10, 1)
    assert len(gamma) == config["CRPS_dim"]
    assert beta[0].shape == (10, model.M + 1)
    assert len(beta) == config["CRPS_dim"]
    assert logit.shape[1] == config["softmax_dim"]
    
    print("Model pass test!")
#%%
if __name__ == '__main__':
    main()
#%%