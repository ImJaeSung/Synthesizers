"""
Reference:
[1] https://github.com/an-seunghwan/DistVAE/blob/main/modules/train.py
"""

#%%
import tqdm

import torch
from torch import nn
#%%
def train_VAE(
    EncodedInfo_list,
    train_dataloader,
    model,
    config,
    optimizer,
    device):

    logs = {
        'loss': [], 
        'quantile': [],
        'KL': [],
    }

    # for debugging
    logs['activated'] = []
    
    for (x_batch) in tqdm.tqdm(iter(train_dataloader), desc="inner loop"):
        
        x_batch = x_batch.to(device)
         
        optimizer.zero_grad()
        
        _, mean, logvar, gamma, beta, logit = model(x_batch)
        
        loss_ = []
        
        """alpha_tilde"""
        alpha_tilde_list = model.quantile_inverse(x_batch, gamma, beta)
        
        """loss"""
        j = 0
        st = 0
        total_loss = 0

        for j, info in enumerate(EncodedInfo_list):
            if info.activation_fn == "CRPS":
                term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde_list[j], model.delta).pow(2)
                term += 2 * torch.maximum(alpha_tilde_list[j], model.delta) * model.delta
                
                loss = (2 * alpha_tilde_list[j]) * x_batch[:, [j]]
                loss += (1 - 2 * alpha_tilde_list[j]) * gamma[j]
                loss += (beta[j] * term).sum(axis=1, keepdims=True)
                loss *= 0.5
                total_loss += loss.mean()
            
            elif info.activation_fn == "softmax":
                ed = st + info.dim
                _, targets = x_batch[:, config["CRPS_dim"] + st : config["CRPS_dim"] + ed].max(dim=1)
                out = logit[:, st : ed]
                total_loss += nn.CrossEntropyLoss()(out, targets)
                st = ed
                
        loss_.append(('quantile', total_loss))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["latent_dim"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        ### activated: for debugging
        var_ = torch.exp(logvar).mean(axis=0)
        loss_.append(('activated', (var_ < 0.1).sum()))
        
        loss = total_loss + config["beta"] * KL 
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%