#%%
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
#%%
def train_function(
    EncodedInfo_list,
    train_datalooper_Cont,
    train_datalooper_Disc, 
    Trainer_Cont,
    Trainer_Disc,
    tabularUnet_Cont,
    config, 
    optimizer_Cont,
    optimizer_Disc,
    scheduler_Cont,
    scheduler_Disc, 
    device):
    
    for epoch in range(config["epochs"]):
        logs = {
            "diffusion_Cont": [],
            "diffusion_Disc": [],
            "triplet_Cont": [],
            "triplet_Disc": [],
            "Cont_loss": [],
            "Disc_loss": [],
        }

        # for x_batch in tqdm(dataloader, desc="inner loop..."):
        x_0_Cont = next(train_datalooper_Cont).to(device).float()
        x_0_Disc = next(train_datalooper_Disc).to(device)

        loss_ = []
        
        optimizer_Cont.zero_grad()
        optimizer_Disc.zero_grad()
        
        negative_sample_Cont, negative_sample_Disc = make_negative_condition(x_0_Cont, x_0_Disc)
        
        timestep = torch.randint(
            config["diffusion_steps"], size=(x_0_Cont.shape[0], ), device=x_0_Cont.device)
        p_timestep = torch.ones_like(timestep).float() / config["diffusion_steps"]
        
        """co-evolving training and predict positive samples"""
        # continuous
        noise = torch.randn_like(x_0_Cont)
        x_t_Cont = Trainer_Cont.make_x_t(x_0_Cont, timestep, noise)
        
        # discrete
        log_x_start = torch.log(x_0_Disc.float().clamp(min=1e-30))
        x_t_Disc = Trainer_Disc.q_sample(log_x_start=log_x_start, timestep=timestep)
        
        # continuous loss
        eps = Trainer_Cont.tabularUnet(x_t_Cont, timestep, x_t_Disc.to(x_t_Cont.device))
        positive_sample_0_Cont = Trainer_Cont.predict_xstart_from_eps(x_t_Cont, timestep, eps=eps)
        diffusion_Cont = F.mse_loss(eps, noise, reduction='none')
        diffusion_Cont = diffusion_Cont.mean()
        loss_.append(("diffusion_Cont", diffusion_Cont))
        
        # discrete loss
        kl, positive_sample_0_Disc = Trainer_Disc.compute_Lt(
            log_x_start, x_t_Disc, timestep, x_t_Cont)
        positive_sample_0_Disc = torch.exp(positive_sample_0_Disc)
        kl_prior = Trainer_Disc.kl_prior(log_x_start)
        diffusion_Disc = (kl / p_timestep + kl_prior).mean()
        loss_.append(("diffusion_Disc", diffusion_Disc))
        
        """negative condition and predict negative samples"""
        # continuous
        noise_negative_sample = torch.randn_like(negative_sample_Cont)
        negative_sample_t_Cont = Trainer_Cont.make_x_t(negative_sample_Cont, timestep, noise_negative_sample)
        
        # discrete
        log_negative_sample_start = torch.log(negative_sample_Disc.float().clamp(min=1e-30))
        negative_sample_t_Disc = Trainer_Disc.q_sample(
            log_x_start=log_negative_sample_start, timestep=timestep)
        
        # continuous
        eps_negative_sample = Trainer_Cont.tabularUnet(
            x_t_Cont, timestep, negative_sample_t_Disc.to(negative_sample_t_Disc.device))
        negative_sample_0_Cont = Trainer_Cont.predict_xstart_from_eps(
            x_t_Cont, timestep, eps=eps_negative_sample)
        
        # discrete
        _, negative_sample_0_Disc = Trainer_Disc.compute_Lt(
            log_x_start, x_t_Disc, timestep, negative_sample_t_Cont)
        negative_sample_0_Disc = torch.exp(negative_sample_0_Disc)
        
        """Contrastive learning"""
        # continuous
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        triplet_Cont = triplet_loss(x_0_Cont, positive_sample_0_Cont, negative_sample_0_Cont)
        loss_.append(("triplet_Cont", triplet_Cont))
        
        # discrete
        st = 0
        tmp_triplet_Disc = []
        for column_info in EncodedInfo_list:
            if column_info[1] == 'softmax':
                ed = st + column_info[0]  
                CE_positive_Disc = F.cross_entropy(
                    positive_sample_0_Disc[:, st:ed], 
                    torch.argmax(x_0_Disc[:, st:ed], dim=-1).long(), 
                    reduction='none')
                
                CE_negative_Disc = F.cross_entropy(
                    negative_sample_0_Disc[:, st:ed],
                    torch.argmax(x_0_Disc[:, st:ed], dim=-1).long(), 
                    reduction='none')
                
                tmp_triplet_Disc.append(
                    max((CE_positive_Disc-CE_negative_Disc).mean() + 1, 0)
                )
                st = ed
        
        triplet_Disc = sum(tmp_triplet_Disc)/len(tmp_triplet_Disc)
        loss_.append(("triplet_Disc", triplet_Disc))
        
        Cont_loss = diffusion_Cont + config["Cont_lambda"] * triplet_Cont
        Disc_loss = diffusion_Disc + config["Disc_lambda"] * triplet_Disc
        loss_.append(("Cont_loss", Cont_loss))
        loss_.append(("Disc_loss", Disc_loss))

        Cont_loss.backward()
        nn.utils.clip_grad_norm_(tabularUnet_Cont.parameters(), config["grad_clip"])
        optimizer_Cont.step()
        scheduler_Cont.step()
        
        Disc_loss.backward()
        nn.utils.clip_grad_value_(Trainer_Disc.parameters(), config["grad_clip"])
        nn.utils.clip_grad_norm_(Trainer_Disc.parameters(), config["grad_clip"])
        optimizer_Disc.step()
        scheduler_Disc.step()

        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
                
        if epoch % 10 == 0:                
            print_input = "[epoch {:03d}]".format(epoch + 1)
            print_input += "".join(
                [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
            )
            print(print_input)

        """update log"""
        wandb.log({x: np.mean(y) for x, y in logs.items()})

    return

#%%
def make_negative_condition(x_0_Cont, x_0_Disc):
    device = x_0_Cont.device
    x_0_Cont = x_0_Cont.detach().cpu().numpy()
    x_0_Disc = x_0_Disc.detach().cpu().numpy()

    raw_negative_sample_Cont = pd.DataFrame(x_0_Cont)
    raw_negative_sample_Disc = pd.DataFrame(x_0_Disc)
    
    negative_sample_Cont = np.array(
        raw_negative_sample_Cont.sample(frac=1, replace = False).reset_index(drop=True))
    negative_sample_Disc = np.array(
        raw_negative_sample_Disc.sample(frac=1, replace = False).reset_index(drop=True))

    return torch.tensor(negative_sample_Cont).to(device), torch.tensor(negative_sample_Disc).to(device)
# %%
