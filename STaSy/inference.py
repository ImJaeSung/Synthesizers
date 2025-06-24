#%%
import os
import time

import pandas as pd
import numpy as np
from scipy.special import softmax
import argparse
import importlib
from dython.nominal import associations

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import utils as mutils
from models.utils import get_data_inverse_scaler
from models.ema import ExponentialMovingAverage
import losses as losses
from utils import save_checkpoint, restore_checkpoint, apply_activate
import sde_lib as sde_lib
import sampling

from synthetic_eval import evaluation
# from modules.model1 import Encoder_model, Decoder_model

from models.utils import set_random_seed
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "2stage_basline" # put your WANDB project name
# entity = "" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["inference"], # put tags of this python project
)
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    ### default configs
    parser.add_argument('--ver', type=int, default=0, 
                        help='ver for repeatable results')
    parser.add_argument("--model", type=str, default="STaSy")
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        Tabular dataset options: 
                        banknote, whitewine, breast, bankruptcy, musk,
                        abalone, anuran, shoppers, default, magic
                        """)
    parser.add_argument('--test_size', type=float, default=0.2)
    
    ### data
    parser.add_argument('--data.image_size',type = int, default=77)
    parser.add_argument('--data.centered', type=bool, default=False, help='Data centered')

    ### model
    parser.add_argument('--model.name', type=str, default="ncsnpp_tabular")
    parser.add_argument('--model.layer_type', type=str, default="concatsquash")
    parser.add_argument('--model.scale_by_sigma', type=bool, default=False)
    parser.add_argument('--model.ema_rate', type=float, default=0.9999)
    parser.add_argument('--model.activation', type=str, default="elu")
    parser.add_argument('--model.nf', type=int, default=64)
    parser.add_argument('--model.hidden_dims', type=list, default=[1024, 2048, 1024, 1024])
    parser.add_argument('--model.conditional', type=bool, default=True)
    parser.add_argument('--model.embedding_type', type=str, default="fourier")
    parser.add_argument('--model.fourier_scale', type=int, default=16)
    parser.add_argument('--model.conv_size', type=int, default=3)

    parser.add_argument('--model.sigma_min', type=float, default=0.01, help='Minimum sigma')
    parser.add_argument('--model.sigma_max', type=float, default=10., help='Maximum sigma')
    parser.add_argument('--model.num_scales', type=int, default=50)
    parser.add_argument('--model.alpha0', type=float, default=0.3)
    parser.add_argument('--model.beta0', type=float, default=0.95)
    ### test
    parser.add_argument('--test.n_iter', type=int, default=1)

    ### optim
    parser.add_argument('--optim.lr', type=float, default=2e-3)
    parser.add_argument('--optim.eps', type=float, default=1e-8, help='Epsilon value')
    parser.add_argument('--optim.beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--optim.weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--optim.optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--optim.warmup', type=int, default=5000)
    parser.add_argument('--optim.grad_clip', type=float, default=1.0)
    
    ### training
    ## default
    parser.add_argument('--training.epoch', type=int, default=10000) #10000 for paper setting
    parser.add_argument('--training.snapshot_freq', type=int, default=300)
    parser.add_argument('--training.eval_freq', type=int, default=100)
    parser.add_argument('--training.snapshot_freq_for_preemption', type=int, default=100)
    parser.add_argument('--training.snapshot_sampling', type=bool, default=True)
    parser.add_argument('--training.likelihood_weighting', type=bool, default=False)
    parser.add_argument('--training.continuous', type=bool, default=True)
    parser.add_argument('--training.eps', type=float, default=1e-05)
    parser.add_argument('--training.loss_weighting', type=bool, default=False)
    parser.add_argument('--training.spl', type=bool, default=True)
    parser.add_argument('--training.lambda_', type=float, default=0.5)

    ## specific
    parser.add_argument('--training.sde', type=str, default="vesde")
    parser.add_argument('--training.reduce_mean', type=bool, default=True)
    parser.add_argument('--training.n_iters', type=int, default=100000)
    parser.add_argument('--training.tolerance', type=float, default=1e-03)
    parser.add_argument('--training.hutchinson_type', type=str, default="Rademacher")
    parser.add_argument('--training.retrain_type', type=str, default="median")
    parser.add_argument('--training.batch_size', type=int, default=1000)

    ## sampling
    parser.add_argument('--sampling.method', type=str, default='ode')
    parser.add_argument('--sampling.predictor', type=str, default='euler_maruyama')
    parser.add_argument('--sampling.corrector', type=str, default='none')
    parser.add_argument('--sampling.n_steps_each', type=int, default=1)
    parser.add_argument('--sampling.noise_removal', type=bool, default=True)
    parser.add_argument('--sampling.probability_flow', type=bool, default=False)
    parser.add_argument('--sampling.snr', type=float, default=0.16)

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

def recover_data(syn_num, syn_cat, info):

    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df

#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    """model load"""
    model_name = f"{config['model']}_{config['dataset']}_{config['model.sigma_min']}_{config['model.sigma_max']}_{config['optim.lr']}_{config['model.beta0']}_{config['model.alpha0']}"
    artifact = wandb.use_artifact(
        f"{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]

    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)
    #%%
    """dataset"""
    dataset_module = importlib.import_module(f"dataset.preprocess")
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    
    train_dataset = CustomDataset(
        config,
        train=True)
    test_dataset = CustomDataset(
        config, 
        train=False, 
        cont_scalers=train_dataset.cont_scalers, 
        cat_scalers=train_dataset.cat_scalers)     
   
    cont_scalers = train_dataset.cont_scalers
    cat_scalers = train_dataset.cat_scalers
    train_z = np.concatenate([train_dataset.X_num, train_dataset.X_cat], axis=1)
    config["data.image_size"] = train_z.shape[1] 
    #%%
    """model"""
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config["model.ema_rate"])
    optimizer = losses.get_optimizer(config, score_model.parameters())
    
    # load_state_dict
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)
    # model_dir = f"./assets/models/{model_name}/"
    # model_name = f'{base_name}_{config["ver"]}'
    state = restore_checkpoint(
        model_dir + '/' + model_name, 
        state, 
        config["device"]
    )
    #%%
    # Setup SDEs
    if config["training.sde"].lower() == 'vpsde':
        sde = sde_lib.VPSDE(
            beta_min=config["model.beta_min"], 
            beta_max=config["model.beta_max"], 
            N=config["model.num_scales"] 
        )
        sampling_eps = 1e-3
    elif config["training.sde"].lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(
            beta_min=config["model.beta_min"], 
            beta_max=config["model.beta_max"],
            N=config["model.num_scales"]
        )
        sampling_eps = 1e-3
    elif config["training.sde"].lower() == 'vesde':
        sde = sde_lib.VESDE(
            sigma_min=config["model.sigma_min"], 
            sigma_max=config["model.sigma_max"],
            N=config["model.num_scales"]
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config['training.sde']} unknown.")
        logging.info(score_model)
    
    """number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params = count_parameters(score_model)
    print("the number of parameters", num_params)
    wandb.log({"the number of parameters": num_params})
    #%%
    #TODO: integrate the model
    """Synthetic data generation"""
    sampling_shape = (train_z.shape[0], config["data.image_size"])

    inverse_scaler = get_data_inverse_scaler(config) ##min_max
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    print('Start sampling...')
    start_time = time.time()
    samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)

    task_type = config['info']['task_type']
    num_col_idx = config['info']['num_col_idx']
    cat_col_idx = config['info']['cat_col_idx']
    target_col_idx = config['info']['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)
    
    syn_data_num = samples[:, :n_num_feat].cpu().numpy()
    syn_data_cat = samples[:, n_num_feat:].cpu().numpy()
    predicted_classes = []
    start = 0
    
    for cat_size  in train_dataset.EncodedInfo.num_categories:
        end = start + cat_size
        logits_block = syn_data_cat[:, start:end]  # (B, cat_size)
        probs = softmax(logits_block, axis=1)   
        pred = np.argmax(probs, axis=1)   
        predicted_classes.append(pred.reshape(-1, 1)) # (B, 1)
        start = end
    syn_cat = np.concatenate(predicted_classes, axis=1)

    data_np = np.concatenate([syn_data_num, syn_cat], axis=1)
    syn_df = pd.DataFrame(data_np, columns=train_dataset.features)
    for col, scaler in train_dataset.cont_scalers.items():
        syn_df[[col]] = scaler.inverse_transform(syn_df[[col]])

    syn_df[train_dataset.categorical_features] = syn_df[train_dataset.categorical_features].astype(int)
    syn_df[train_dataset.integer_features] = syn_df[train_dataset.integer_features].round(0).astype(int)
        

    end_time = time.time()

    print(f'Sampling time = {end_time - start_time}')

    #%% 
    results = evaluation.evaluate(
        syn_df, train_dataset.raw_data, test_dataset.raw_data, 
        train_dataset.ClfTarget, train_dataset.continuous_features, train_dataset.categorical_features, device
        )
        
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
        
    print("Pairwise correlation difference (PCD)...")
    syn_asso = associations(
        syn_df, 
        nominal_columns=train_dataset.categorical_features,
        compute_only=True)
    true_asso = associations(
        train_dataset.raw_data, 
        nominal_columns=train_dataset.categorical_features,
        compute_only=True)
    pcd_corr = np.linalg.norm(true_asso["corr"] - syn_asso["corr"])
    print("Pairwise correlation difference (PCD) : ",pcd_corr)
    wandb.log({"PCD":pcd_corr})

    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%
    
    