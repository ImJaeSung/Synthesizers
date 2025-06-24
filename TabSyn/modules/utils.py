#%% Utils for DIffusion model
"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""
import importlib
import torch
import numpy as np
import random
from scipy.stats import betaprime

import os
import json
import pandas as pd
from modules.model1 import Decoder_model 

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
#%%
#================================Mine========================================
def make_tensor(x):
    if isinstance(x, str):
        x = x.strip("[]")   
        arr = np.fromstring(x, sep=" ")
        return torch.from_numpy(arr).float()
    elif isinstance(x, list):  
        # 만약 메타데이터에서 리스트 형태로 넘어왔다면
        return torch.tensor(x).float()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        return x.float()
    else:
        # 그 외(예: int, float 등) → 텐서로 변환
        return torch.tensor(x).float()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)   

randn_like=torch.randn_like

SIGMA_MIN=0.002
SIGMA_MAX=80
rho=7
S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

def recover_data(syn_num, syn_cat, syn_target, info):
        raw_df = np.concatenate([syn_num,syn_cat,syn_target],axis=1)
        raw_df.shape
        num_col_idx = info['num_col_idx']
        cat_col_idx = [col for col in info['cat_col_idx'] if col not in info['target_col_idx']]
        target_col_idx = info['target_col_idx']

        idx_mapping = info['idx_mapping']
        idx_mapping = {int(key): value for key, value in idx_mapping.items()}
        print("idx_mapping",idx_mapping)
        syn_df = pd.DataFrame()
   
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = raw_df[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = raw_df[:, idx_mapping[i]]
            else:
                syn_df[i] = raw_df[:,idx_mapping[i]]

        return syn_df

def sample(net, num_samples, dim, num_steps = 50, device = 'cuda:0'):
    latents = torch.randn([num_samples, dim], device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next

def sample_step(net, num_steps, i, t_cur, t_next, x_next):

    x_cur = x_next
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur) 
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    # Euler step.

    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, denosie_fn, data, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(self, denosie_fn, data, labels = None, augment_pipe=None, stf=False, pfgmpp=False, ref_data=None):
        if pfgmpp:

            # N, 
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels,  augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim = 100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts


    def __call__(self, denoise_fn, data):

        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma)
    
        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss


def get_input_train(args):
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/'
    embedding_save_path = f'{curr_dir}/vae/ckpt/{dataname}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    return train_z, curr_dir, dataset_dir, ckpt_dir, info

@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, config, device):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    print("number of numerical features",n_num_feat)

    model_module = importlib.import_module('modules.model1')
    importlib.reload(model_module)
    ### MODEL INIT
    vae = model_module.Model_VAE(
        config['num_layers'],  
        config["d_numerical"], 
        config['categories'], 
        config['d_token'],
        factor=config["factor"])
    pre_decoder = Decoder_model(
        num_layers=config["num_layers"],
        d_numerical=config["d_numerical"],
        categories=config["categories"],
        d_token=config["d_token"],
        n_head=config["n_head"],
        factor=config["factor"]
    )
    vae.load_state_dict(torch.load(info["model_dir"]))
    vae.eval()
    pre_decoder.load_weights(vae)
    pre_decoder.eval()
    token_dim = config['token_dim']

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)
    # train_dataset.category_maps
    norm_input = pre_decoder(torch.tensor(syn_data))
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim = -1))

    syn_num = x_hat_num.cpu().detach().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()
    
    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)
    print("len(target_col_idx)",len(target_col_idx))
    syn_target = syn_cat[:, len(cat_col_idx)-1:]
    syn_cat = syn_cat[:, :len(cat_col_idx)-1]

    return syn_num, syn_cat, syn_target

def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat
    
def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping

def memorization_ratio(train, syndata, continuous_features, categorical_features):
    ### pre-processing
    train_ = train.copy()
    syndata_ = syndata.copy()
    # continuous: standardization
    scaler = StandardScaler().fit(train_[continuous_features])
    train_[continuous_features] = scaler.transform(train_[continuous_features])
    syndata_[continuous_features] = scaler.transform(syndata_[continuous_features])
    # categorical: one-hot encoding
    scaler = OneHotEncoder(handle_unknown='ignore').fit(train_[categorical_features])
    train_ = np.concatenate([
        train_[continuous_features].values,
        scaler.transform(train_[categorical_features]).toarray()
    ], axis=1)
    syndata_ = np.concatenate([
        syndata_[continuous_features].values,
        scaler.transform(syndata_[categorical_features]).toarray()
    ], axis=1)

    dist = metrics.pairwise_distances(syndata_, Y=train_, n_jobs=-1)
    dist = np.sort(dist, axis=1)
    EPS = 1e-8
    ratio = dist[:, 0] / (dist[:, 1] + EPS)
    
    return ratio 