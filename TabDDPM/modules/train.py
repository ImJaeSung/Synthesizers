#%%
from tqdm import tqdm
from copy import deepcopy
import torch
import os
import numpy as np
# import zero

import wandb
import lib
import pandas as pd
from modules.utils import update_ema

#%%
class Trainer:
    def __init__(
        self, 
        diffusion, 
        train_dataloader, 
        config, 
        device
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_dataloader = train_dataloader
        self.steps = config["steps"]
        self.init_lr = config["lr"]
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 10
        # self.print_every = 100
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            logs = {
                'Mloss': [], 
                'Gloss': [], 
                'loss': [], 
            }
            
            for x, out_dict in tqdm(self.train_dataloader, desc="inner loop..."):
                
                loss = []

                out_dict = {'y': out_dict}
                batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

                self._anneal_lr(step)

                curr_count += len(x)
                curr_loss_multi += batch_loss_multi.item() * len(x)
                curr_loss_gauss += batch_loss_gauss.item() * len(x)

                if (step + 1) % self.log_every == 0:
                    mloss = np.around(curr_loss_multi / curr_count, 4)
                    gloss = np.around(curr_loss_gauss / curr_count, 4)
                    
                    # if (step + 1) % self.print_every == 0:
                        # print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                    print_input = f'Step {(step + 1)}/{self.steps}'
                    print_input += ''.join([f'MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}'])
                    print(print_input)

                    loss.append(('Mloss', mloss))
                    loss.append(('GLoss', gloss))
                    loss.append(('loss', mloss + gloss))        
                            
                    curr_count = 0
                    curr_loss_gauss = 0.0
                    curr_loss_multi = 0.0

                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())
                step += 1
                    
            """update log"""
            wandb.log({x : np.mean(y) for x, y in logs.items()})
        
        return
    #%%


    # trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    # torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    # torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))

