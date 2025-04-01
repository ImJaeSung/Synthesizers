#%%
from tqdm import tqdm
import numpy as np

import torch
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
            mask = torch.rand_like(batch, dtype=torch.float32, device=device) > torch.rand(len(batch), 1, device=device)
            mask = mask.to(device)
            
            loss_ = []

            optimizer.zero_grad()

            pred = model(batch, mask)  # (feature, batch, category)
            
            loss = 0.0
            for j in range(model.EncodedInfo.num_features):
                tmp_loss = F.cross_entropy(
                    pred[j][mask[:, j]],
                    batch[:, j][mask[:, j]]
                )  # ignore unmasked

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

