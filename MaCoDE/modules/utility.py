#%%
import torch

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)   
#%%
def marginal_plot(train, syndata, config):
    if not os.path.exists(f"./assets/figs/{config['dataset']}/seed{config['seed']}/"):
        os.makedirs(f"./assets/figs/{config['dataset']}/seed{config['seed']}/")
    
    figs = []
    for idx, feature in tqdm(enumerate(train.columns), desc="Plotting Histograms..."):
        fig = plt.figure(figsize=(7, 4))
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=syndata,
            x=syndata[feature],
            stat='density',
            label='synthetic',
            ax=ax,
            bins=int(np.sqrt(len(syndata)))) 
        sns.histplot(
            data=train,
            x=train[feature],
            stat='density',
            label='train',
            ax=ax,
            bins=int(np.sqrt(len(train)))) 
        ax.legend()
        ax.set_title(f'{feature}', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"./assets/figs/{config['dataset']}/seed{config['seed']}/hist_{feature}.png")
        # plt.show()
        plt.close()
        figs.append(fig)
    return figs
#%%