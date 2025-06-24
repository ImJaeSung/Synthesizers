#%%
import numpy as np
import torch
import torch.nn as nn
import wandb 

from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time

from modules.model1 import Encoder_model, Decoder_model

warnings.filterwarnings('ignore')

# LR = 1e-3
# WD = 0 ##weight decay
# D_TOKEN = 4
# TOKEN_BIAS = True

# N_HEAD = 1
# FACTOR = 32
# NUM_LAYERS = 2

def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0
    
    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc

def train_function(model, train_dataset, valid_dataset, train_dataloader, config, optimizer,scheduler, device):
    model.train()
    
    """configuration"""
    num_epochs = config['epochs']
    max_beta = config['max_beta']
    min_beta = config['min_beta']
    lambd = config['lambda']  ### Becareful "lambd"
    beta = max_beta ### beta initialize
    best_train_loss = float('inf')
    current_lr = optimizer.param_groups[0]['lr']

    X_train_num = train_dataset.X_num #14
    X_train_cat = train_dataset.X_cat #10
    X_test_num = valid_dataset.X_num
    X_test_cat = valid_dataset.X_cat
    
    """Saving model information"""
    base_name = f"{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    base_name += f"_{config['batch_size1']}_{config['batch_size2']}_{config['max_beta']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"

    """model initialize"""
    pre_encoder = Encoder_model(
        num_layers=config["num_layers"],
        d_numerical=config["d_numerical"],
        categories=config["categories"],
        d_token=config["d_token"],
        n_head=config["n_head"],
        factor=config["factor"]
    ).to(device)
    pre_decoder = Decoder_model(
        num_layers=config["num_layers"],
        d_numerical=config["d_numerical"],
        categories=config["categories"],
        d_token=config["d_token"],
        n_head=config["n_head"],
        factor=config["factor"]
    ).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    ###
    patience = 0
    for epoch in range(num_epochs):
        logs = {
            'loss' : [],
            'loss_mse': [],
            'loss_ce': [],
            'loss_kld': [],
            'train_acc': [],
            'val_loss' : [],
            'val_ce' : [],
            'val_acc' : []
        }
        model.train()
        for batch_num, batch_cat in train_dataloader:
            loss_ = []
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
        
            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(
                batch_num, batch_cat, 
                Recon_X_num, Recon_X_cat, 
                mu_z, std_z
            )
            if config['d_numerical'] == 0:
                loss_mse = torch.tensor([0], device=device)
            
            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()
            loss_.append(('loss', loss))
            loss_.append(('loss_mse', loss_mse))
            loss_.append(('loss_ce', loss_ce))   
            loss_.append(('loss_kld', loss_kld))   
            loss_.append(('train_acc', train_acc))   
        
        '''
            Validation
        '''
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(
                torch.from_numpy(X_test_num).float().to(device),
                torch.from_numpy(X_test_cat).long().to(device))

            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(
                torch.from_numpy(X_test_num).float().to(device),
                torch.from_numpy(X_test_cat).long().to(device),
                Recon_X_num,
                Recon_X_cat,
                mu_z,
                std_z
            )
            if config['d_numerical'] == 0:
                val_mse_loss = torch.tensor([0], device=device)
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()    

            logs["val_loss"].append(val_loss)
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")
                
            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), f"./{model_dir}/stage1_{model_name}.pth") ### stage 1
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd
            logs['val_ce'].append(val_ce_loss.item())
            logs['val_acc'].append(val_acc.item())
        
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]

        if (epoch + 1) % 100 == 0:
            print_input = "[epoch {:03d}]".format(epoch + 1)
            print_input += ''.join([', {}: {:.4f}'.format(x, np.sum(y)) for x, y in logs.items()])
            print(print_input)

        wandb.log({x : np.sum(y) for x, y in logs.items()}) 
        

    with torch.no_grad():
        ### saving the latent embeddings
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)  

        X_train_num = torch.from_numpy(X_train_num).float().to(device)
        X_train_cat = torch.from_numpy(X_train_cat).long().to(device)
        
        print('Successfully load and save the Enc/Dec models!')
        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
        np.save(f'{model_dir}/stage1_{model_name}_train_z.npy', train_z)

        print('Successfully made pretrained embeddings : "train_z"')
    

    return train_z