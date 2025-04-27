#%%
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch import nn
import torch.nn.functional as F
#%%
def multiclass_loss(model, batch, pred, mask):
    class_loss = 0.
    for j in range(model.EncodedInfo.num_features):
        tmp = F.cross_entropy(
            pred[j][mask[:, j]], # ignore [MASKED] token probability
            batch[:, j][mask[:, j]].long()-1 # ignore unmasked
        )
        if not tmp.isnan():
            class_loss += tmp
    return class_loss
#%%
def ranked_probability_score(model,batch, pred):
    RPS = 0.
    for j in range(model.EncodedInfo.num_features):
        CDF = pred[j].softmax(dim=1).cumsum(dim=1)
        target = F.one_hot(batch[:, j].long(), num_classes=pred[j].size(1)).cumsum(dim=1)
        RPS += (CDF - target).pow(2).sum(dim=1).mean()
    return RPS
#%%
def train_function(
    model,
    config,
    optimizer, 
    train_dataloader,
    device):
    
    for epoch in tqdm(range(config["epochs"]), desc="Training..."):
        logs = {}
        
        for batch in train_dataloader:
            batch = batch.to(device)
            
            mask = torch.rand(batch.size(0), model.EncodedInfo.num_features) > torch.rand(len(batch), 1)
            mask = mask.to(device)
            
            masked_batch = batch.clone()
            masked_batch[mask] = 0. # [MASKED] token
            
            loss_ = []
            
            optimizer.zero_grad()
            
            pred = model(masked_batch)
            
            if config['loss'] == 'multiclass'":
                loss = multiclass_loss(model, batch, pred, mask)
            
            elif config['loss'] == 'RPS':
                loss = ranked_probability_score(batch, )
        
        
            loss_.append(('loss', loss))
            
            loss.backward()
            optimizer.step()
            
            for x, y in loss_:
                try:
                    logs[x] = logs.get(x) + [y.item()]
                except:
                    logs[x] = []
                    logs[x] = logs.get(x) + [y.item()]

        if epoch % 10 == 0:
            print_input = "[epoch {:03d}]".format(epoch + 1)
            print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
            print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})

    return
#%%
# def categorical_loss(model, batch, pred, mask):
#     cat_loss = 0.
#     for j in range(model.EncodedInfo.num_features):
#         if j >= model.EncodedInfo.num_continuous_features:
#             tmp = F.cross_entropy(
#                 pred[j][mask[:, j]], # ignore [MASKED] token probability
#                 batch[:, j][mask[:, j]].long()-1 # ignore unmasked
#             )
#             if not tmp.isnan():
#                 cat_loss += tmp
#     return cat_loss

# def ordinal_regression_loss(model, batch, pred, mask):
#     ordinal_loss = 0.
#     for j in range(model.EncodedInfo.num_continuous_features):
#         logit = nn.Softplus()(pred[j][mask[:, j]]).cumsum(dim=1)
#         logit = logit - logit[:, [logit.size(1)//2]]
#         cumprobs = nn.Sigmoid()(logit / logit.size(1))
#         upper = torch.cat([cumprobs, torch.ones((cumprobs.size(0), 1)).to(model.device)], dim=1)
#         lower = torch.cat([torch.zeros((cumprobs.size(0), 1)).to(model.device), cumprobs], dim=1)
#         probs = upper - lower
#         tmp = F.cross_entropy(
#             probs, # ignore [MASKED] token probability
#             batch[:, j][mask[:, j]].long()-1 # ignore unmasked
#         )
#         if not tmp.isnan():
#             ordinal_loss += tmp
#     return ordinal_loss

# def JointBCE(model, batch, pred, mask):
#     ordinal_loss = 0.
#     tril = torch.tril(torch.ones(model.config["bins"], model.config["bins"])).to(model.device)
#     for j in range(model.EncodedInfo.num_continuous_features):
#         CDF = pred[j][mask[:, j]].softmax(dim=1).cumsum(dim=1) # numerical stability
#         CDF = torch.clip(CDF, min=0., max=1.) # numerical stability
#         label = tril[:, batch[:, j][mask[:, j]].long()-1].t().to(model.device)
#         tmp = F.binary_cross_entropy(CDF, label)
#         if not tmp.isnan():
#             ordinal_loss += tmp
#     return ordinal_loss
#%%