#%%
import os
import argparse
import importlib

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.utils import set_random_seed
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

project = "2stage_baseline" # put your WANDB project name
# entity = "" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument("--model", type=str, default="TabSyn")
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        Tabular dataset options: 
                        banknote, whitewine, breast, bankruptcy, musk,
                        abalone, anuran, shoppers, default, magic
                        """)
    parser.add_argument('--test_size', default=0.2, type=float, 
                        help="Proportion of the dataset to include in the test split")
    
    ### VAE model
    parser.add_argument('--batch_size1', default=1024, type=int, 
                        help="Batch size for 1st training")
    parser.add_argument('--lr1', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay1', default=0, type=int,
                        help='weight decay for model 1')
    parser.add_argument('--beta', default=1, type=float,
                        help='beta for model 1')
    parser.add_argument('--d_token', default=4, type=int,
                        help='tokenizer dimension, the small d')
    parser.add_argument('--factor', default=32, type=int,
                        help='FACTOR')
    parser.add_argument('--TOKEN_BIAS', default=True, type=bool,
                        help = 'Token Bias')
    parser.add_argument('--n_head', default=1, type=int,
                        help='N_HEAD')
    parser.add_argument('--bias', default=True, type=bool,
                        help='Token Bias')
    parser.add_argument('--num_layers', default=2, type=int,
                        help='the number of layer in transformer')
    parser.add_argument('--epochs', default=4000, type=int,
                        help='epochs') #4000 for default value

    # configs for traing TabSyn's VAE
    ### Max/Min optimal beta finding is needed via Exp.
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Maximum beta')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum beta.')
    parser.add_argument('--lambda', type=float, default=0.7, help='beta annealing')

    ### Diffusion Model
    parser.add_argument('--batch_size2', default=1024, type=int,
                        help='batch size for 2stage : fixed')
    parser.add_argument('--scheduler', type=str, default='linear', 
                        help="options for beta scheduling: linear and cosine")
    parser.add_argument("--num_timesteps", type=int, default=1000, 
                        help="the number of timesteps")
    parser.add_argument('--weight_decay2', default=1e-4, type=float,
                        help='weight decay for AdamW')
    parser.add_argument('--epochs_2', default=10_000, type=int,
                        help='epochs for model 2 : fixed') #10_000
    parser.add_argument("--denoising_dim", default=1024, type=int,
                        help="fixed for 1024.")
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    wandb.config.update(config)
    
    """Dataset"""
    dataset_module = importlib.import_module(f"datasets.preprocess")
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(
        config, train=True)
    valid_dataset = CustomDataset(
        config, train='valid', cont_scalers=train_dataset.cont_scalers, cat_scalers=train_dataset.cat_scalers)     
    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size1'], shuffle=True)
    
    config["d_numerical"] = train_dataset.num_continuous_features
    config['categories'] = train_dataset.num_categories # the number of unique value
    #%% 
    """Model 1 (VAE)"""
    model_module = importlib.import_module('modules.model1')
    importlib.reload(model_module)
    model1 = model_module.Model_VAE(
        num_layers=config['num_layers'],  
        d_numerical=config["d_numerical"], 
        categories=config['categories'], 
        d_token=config['d_token'],
        factor=config["factor"],
        n_head=config["n_head"],
        bias=config['bias']
    ).to(device) 
    
    optimizer1 = torch.optim.Adam(
        model1.parameters(), 
        lr=config['lr1'],
        weight_decay=config["weight_decay1"]
    )
    scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.95, patience=10)
    #%% 
    """Train (Stage 1)"""
    train_module1 = importlib.import_module('modules.train1')
    importlib.reload(train_module1)
    train_z = train_module1.train_function( 
        model1,
        train_dataset,
        valid_dataset,
        train_dataloader,
        config,
        optimizer1, 
        scheduler1,
        device)

    train_z = torch.tensor(train_z).float()[:,1:,:]
    B, num_tokens, token_dim = train_z.size()
    train_z = train_z.view(B, num_tokens * token_dim)
    print('train_z is formulated by size:',train_z.shape)
    #%% 
    """Model 2 (Diffusion)""" 
    denoise_fn_module = importlib.import_module(f"modules.model2")
    importlib.reload(denoise_fn_module)
    denoise_fn = denoise_fn_module.MLPDiffusion(
        num_tokens * token_dim,
        config['denoising_dim'], ## fixed 1024
    ).to(device)
    denoise_fn.train()
    
    model2 = denoise_fn_module.Model(
        denoise_fn=denoise_fn, 
        hid_dim=num_tokens*token_dim
    ).to(device)
    #%%
    """the number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params_1 = count_parameters(model1) 
    num_params_2 = count_parameters(model2)
    print(f"Number of VAE Parameters: {num_params_1 / 1_000_000:.6f}M")
    print(f"Number of Diffusion Parameters: {num_params_2 / 1_000_000:.6f}M")
    wandb.log({"Number of VAE Parameters": num_params_1 / 1000000})
    wandb.log({"Number of Diffusion Parameters": num_params_2 / 1000000})
    ##%
    """training"""
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay=0) ## fixed
    scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.9, patience=20)

    train_module2 = importlib.import_module('modules.train2')
    importlib.reload(train_module2)
    train_module2.train_function(
        config,
        model2,
        train_z,
        optimizer2,
        scheduler2,
        device,
    )
    ##%
    """model save"""
    base_name = f"TabSyn_{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    base_name += f"_{config['batch_size1']}_{config['batch_size2']}_{config['max_beta']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"
    torch.save(model2.state_dict(), f"./{model_dir}/{model_name}.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config) 
    
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./datasets/preprocess.py')
    artifact.add_file('./modules/train1.py')
    artifact.add_file('./modules/model1.py')
    artifact.add_file('./modules/train2.py')
    artifact.add_file('./modules/model2.py')
    artifact.add_file('./main.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()