import numpy as np
import random

import torch

#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대한 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NumPy 시드 고정
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