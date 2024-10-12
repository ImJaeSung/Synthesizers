#%%
import torch
import torch.nn as nn
import numpy as np

import wandb
from tqdm import tqdm

#%%
class GoggleLoss(nn.Module):
    def __init__(self, alpha=1, beta=0, graph_prior=None, device="cpu"):
        super(GoggleLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.device = device
        self.alpha = alpha
        self.beta = beta
        if graph_prior is not None:
            self.use_prior = True
            self.graph_prior = (
                torch.Tensor(graph_prior).requires_grad_(False).to(device)
            )
        else:
            self.use_prior = False

    def forward(self, x_recon, x, mu, logvar, graph):
        loss_mse = self.mse_loss(x_recon, x)
        loss_kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if self.use_prior:
            loss_graph = (graph - self.graph_prior).norm(p=1) / torch.numel(graph)
        else:
            loss_graph = graph.norm(p=1) / torch.numel(graph)

        loss = loss_mse + self.alpha * loss_kld + self.beta * loss_graph

        return loss, loss_mse, loss_kld, loss_graph
#%%
def train_function_iter(
    train_dataloader, 
    model,
    optimizer_ga, 
    optimizer_gl, 
    config, 
    device):
    
    """iterative optimization"""
    loss_function = GoggleLoss(
        alpha=config["alpha"],
        beta=config["beta"],
        graph_prior=config["graph_prior"],
        device=device
    )
    
    for epoch in range(config["epochs"]):
        logs = {
            'loss':[],
            'mse':[],
            'kl':[],
            'graph':[]
        }
    
        for i, batch in tqdm(enumerate(train_dataloader), desc="inner loop..."):
            loss_ = []
            batch = batch.to(device)
            if i % 2 == 0:
                optimizer_ga.zero_grad()
                
                x_hat, adj, mu_z, logvar_z = model(batch, epoch)
                loss, mse, kl, graph = loss_function(
                    x_hat, batch, mu_z, logvar_z, adj)

                loss.backward(retain_graph=True)
                optimizer_ga.step()
                loss_.append(('loss', loss))
                loss_.append(('mse', mse))
                loss_.append(('kl', kl))   
                loss_.append(('graph', graph))   

            else:
                optimizer_gl.zero_grad()

                x_hat, adj, mu_z, logvar_z = model(batch, epoch)

                loss, mse, kl, graph = loss_function(
                    x_hat, batch, mu_z, logvar_z, adj)

                loss.backward(retain_graph=True)
                optimizer_gl.step() 
                loss_.append(('loss', loss))
                loss_.append(('mse', mse))
                loss_.append(('kl', kl))   
                loss_.append(('graph', graph))   
            
            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()] 
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.sum(y)) for x, y in logs.items()])
        print(print_input)

        """update log"""
        wandb.log({x : np.sum(y) for x, y in logs.items()}) 
    
    return
#%%
def train_function(
    train_dataloader, 
    model, 
    optimizer,
    config, 
    device):
    
    loss_function = GoggleLoss(
        alpha=config["alpha"],
        beta=config["beta"],
        graph_prior=config["graph_prior"],
        device=device
    )
    
    for epoch in range(config["epochs"]):
        logs = {
            'loss':[],
            'mse':[],
            'kl':[],
            'graph':[]
        }
    
        for i, batch in tqdm(enumerate(train_dataloader), desc="inner loop..."):
            loss_ = []
            batch = batch.to(device)
            optimizer.zero_grad()

            x_hat, adj, mu_z, logvar_z = model(batch, epoch)
            loss, mse, kl, graph = loss_function(x_hat, batch, mu_z, logvar_z, adj)

            loss.backward(retain_graph=True)
            optimizer.step()
            loss_.append(('loss', loss))
            loss_.append(('mse', mse))
            loss_.append(('kl', kl))   
            loss_.append(('graph', graph))   
            
            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()] 
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)

        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()}) 
    return