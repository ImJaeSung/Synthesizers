#%%
import os
import torch
import wandb
import numpy as np

from torch.utils.data import DataLoader
import warnings

from tqdm import tqdm

warnings.filterwarnings('ignore')
#%%

def train_function(config, model, train_z, optimizer, scheduler, device): 
    model.train()
    
    """model save part"""
    base_name = f"{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    base_name += f"_{config['batch_size1']}_{config['batch_size2']}_{config['max_beta']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"

    mean = train_z.mean(0)
    train_z = (train_z - mean) / 2
    
    num_epochs = config['epochs_2']

    train_dataloader = DataLoader(
        train_z,
        batch_size=config["batch_size2"],
        shuffle=True,
        num_workers=4,
    )

    best_loss = float('inf')
    patience = 0
    for epoch in range(num_epochs):
        logs = {
            'diffusion_loss' : [],
        }
        batch_loss = 0.0
        len_input = 0
        for batch in train_dataloader:
            inputs = batch.float().to(device)
            loss = model(inputs)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)
        logs["diffusion_loss"].append(curr_loss)
        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f"./{model_dir}/stage2_{model_name}.pth") ### stage 2
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break
        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f"./{model_dir}/stage2_{model_name}.pth") ### stage 2
        
        if epoch % 500 == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}/{num_epochs}] diffusion_loss: {curr_loss:.6f}")
        
        wandb.log({x : np.mean(y) for x, y in logs.items()})