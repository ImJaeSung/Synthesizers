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
            nan_mask = batch.isnan()
            
            mask1 = torch.rand(batch.size(0), model.EncodedInfo.num_features) > torch.rand(len(batch), 1)
            mask = mask.to(device)
            
            mask = mask1 | nan_mask
            loss_mask = mask1 & ~nan_mask
            
            masked_batch = batch.clone()
            masked_batch[mask] = 0. # [MASKED] token
            
            loss_ = []
            
            optimizer.zero_grad()
            
            pred = model(masked_batch)
            
            loss = multiclass_loss(model, batch, pred, mask)
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