import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb

#%%
def train_function(
        model,
        config, 
        optimizer, 
        scheduler, 
        train_dataloader, 
        device):
    
    for epoch in range(config["epochs"]):
        logs = {
            'loss': []
        }

        for batch in tqdm(train_dataloader, desc="inner roop"):
            batch = batch.to(device)
            mask1 = torch.rand_like(batch, dtype=torch.float32) > torch.rand(len(batch), 1)
            mask1 = mask1.to(device)
            nan_mask = batch == -1
            # nan_mask = nan_mask.to(device)
            
            mask = mask1|nan_mask
            loss_mask = mask1 & ~nan_mask
            
            batch[(batch == -1)] = 0

            loss_ = []

            optimizer.zero_grad()
            
            pred = model(batch, mask) # (feature, batch, category)
            
            loss = 0.0
            for j in range(len(pred)):
                tmp_loss = F.cross_entropy(
                    pred[j][loss_mask[:,j]], 
                    batch[:,j][loss_mask[:,j]]
                ) # ignore unmasked
                
                if not tmp_loss.isnan():
                    loss += tmp_loss

            loss_.append(('loss', loss))
            
            loss.backward()
            optimizer.step()
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]

        scheduler.step()
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})

    return