import numpy as np
import random

import torch

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
def warmup_lr(step):
    return min(step, 5000) / 5000

#%%
def infiniteloop(dataloader):
    while True:
        for _, y in enumerate(dataloader):
            yield y
