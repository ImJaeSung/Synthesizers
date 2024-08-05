"""
Reference:
[1] https://github.com/yandex-research/tab-ddpm/blob/main/scripts/train.py
"""
#%%
from tqdm import tqdm
from copy import deepcopy
import numpy as np

import wandb
from modules.utils import update_ema

#%%
class Trainer:
    def __init__(
        self, 
        diffusion, 
        train_dataloader, 
        optimizer,
        config, 
        device
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.config = config
        self.train_dataloader = train_dataloader
        # self.steps = config["steps"]
        self.init_lr = config["lr"]
        self.optimizer = optimizer
        self.device = device
        # self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        # self.log_every = 11
        # self.print_every = 100
        self.ema_every = 1000

    def _anneal_lr(self, epoch):
        frac_done = epoch / self.config["epochs"]
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def run_loop(self):
        for epoch in range(self.config["epochs"]):
            logs = {
                'Mloss': [], 
                'Gloss': [], 
                'loss': [], 
            }
            
            for x, out_dict in tqdm(self.train_dataloader, desc="inner loop..."):
                
                loss_ = []

                out_dict = {'y': out_dict}
                # batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)
                x = x.to(self.device)
                for k in out_dict:
                    out_dict[k] = out_dict[k].long().to(self.device)

                self.optimizer.zero_grad()
                
                multi_loss, gauss_loss = self.diffusion.mixed_loss(x, out_dict)
                loss = multi_loss + gauss_loss
                loss.backward()
                self.optimizer.step()

                self._anneal_lr(epoch)

                loss_.append(('Mloss', multi_loss))
                loss_.append(('Gloss', gauss_loss))
                loss_.append(('loss', loss))    

                """accumulate losses"""
                for x, y in loss_:
                    logs[x] = logs.get(x) + [y.item()]         

                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            print_input = "[epoch {:03d}]".format(epoch + 1)
            print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
            print(print_input)

            """update log"""
            wandb.log({x : np.mean(y) for x, y in logs.items()})

        return
#%%
