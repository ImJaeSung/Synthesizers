#%%
import os

import numpy as np
import torch

import argparse
import importlib
import sys
from datasets.preprocess import build_num_inverse_fn, build_cat_inverse_fn
from modules.utils import (set_random_seed, 
                           memorization_ratio, 
                           sample,
                           recover_data, 
                           split_num_cat_target)
from synthetic_eval import evaluation

#%%
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "2stage_baseline" # put your WANDB project name
# entity = "" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["memorization"], # put tags of this python project
)

#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    ### defualt
    parser.add_argument('--ver', type=int, default=0, 
                        help='ver for repeatable results')
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
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    model_name = f"{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    model_name += f"_{config['batch_size1']}_{config['batch_size2']}_{config['max_beta']}"

    artifact = wandb.use_artifact(
        f"{project}/TabSyn_{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    info = artifact.metadata.get("info", {})
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """Dataset"""
    dataset_module = importlib.import_module(f"datasets.preprocess")
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    
    train_dataset = CustomDataset(
        config, train=True)
    test_dataset = CustomDataset(
        config, train=False, cont_scalers=train_dataset.cont_scalers, cat_scalers=train_dataset.cat_scalers )        
    
    num_inverse = build_num_inverse_fn(train_dataset.cont_scalers)
    cat_inverse = build_cat_inverse_fn(train_dataset.cat_scalers)
    #%%
    """Model 2 (Diffusion)"""
    train_z = np.load(
        f'./assets/models/{model_name}/stage1_{model_name}_{config["seed"]}_train_z.npy'
    )
    train_z = torch.tensor(train_z).float()
    train_z = train_z[:, 1:, :]
    
    B, num_tokens, token_dim = train_z.size()
    train_z = train_z.view(B, num_tokens * token_dim)
    
    config["token_dim"] = token_dim
    #%%
    denoise_fn_module = importlib.import_module(f"modules.model2")
    importlib.reload(denoise_fn_module)
    denoise_fn = denoise_fn_module.MLPDiffusion(
        num_tokens * token_dim,
        config['denoising_dim'], ## fixed
    ).to(device)
    model2 = denoise_fn_module.Model(
        denoise_fn=denoise_fn, 
        hid_dim =num_tokens * token_dim
    ).to(device) 
    #%%
    if config["cuda"]:
        model2.load_state_dict(
            torch.load(
                # model_dir + f"models/{model_name}/stage2_{model_name}_{config['seed']}.pth"
                f"./assets/models/{model_name}/stage2_{model_name}_{config['seed']}.pth"
            )
        )
    else:
        model2.load_state_dict(
            torch.load(
                # model_dir + f"/stage2_{model_name}.pth",
                f"./{model_dir}/stage2_{model_name}.pth",
                map_location=torch.device("cpu"),
            )
        )
    info['model_dir'] = f"./assets/models/{model_name}/stage1_{model_name}_{config['seed']}.pth"
    model2.eval()
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params_2 = count_parameters(model2)
    print(f"Number of Diffusion Parameters: {num_params_2 / 1_000_000:.6f}M")
    wandb.log({"Number of Diffusion Parameters": num_params_2 / 1000000})
    #%%
    """Synthetic data generation"""
    #TODO: integrate the model code 
    x_next = sample(
        model2.denoise_fn_D, 
        B,
        num_tokens * token_dim
    )
    x_next = x_next * 2 + train_z.mean(0).to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_data = syn_data.astype(np.float32)

    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, info, num_inverse, cat_inverse, config, device) 
    syndata = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    
    syndata.rename(columns = idx_name_mapping, inplace=True)
    syndata[train_dataset.categorical_features] = syndata[train_dataset.categorical_features].astype(int)
    syndata[train_dataset.integer_features] = syndata[train_dataset.integer_features].astype(int)
    syndata[train_dataset.continuous_features] = syndata[train_dataset.continuous_features].astype(np.float32)
    #%%
    """evaluation"""
    results = evaluation.evaluate(
        syndata, 
        train_dataset.raw_data.astype('float32'), 
        test_dataset.raw_data.astype('float32'), 
        train_dataset.ClfTarget, 
        train_dataset.continuous_features, 
        train_dataset.categorical_features, 
        device
    )
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
        
    from dython.nominal import associations
    print("Pairwise correlation difference (PCD)...")
    syn_asso = associations(
        syndata, 
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
    """
    Memorization criterion:
    [1] Diffusion Probabilistic Models Generalize when They Fail to Memorize (Yoon et al., 2023)
    """
    ratio = memorization_ratio(
        train_dataset.raw_data.astype('float32'),  
        syndata, 
        train_dataset.continuous_features, 
        train_dataset.categorical_features
    )
    test_ratio = memorization_ratio(
        train_dataset.raw_data.astype('float32'), 
        test_dataset.raw_data.astype('float32'), 
        train_dataset.continuous_features,
        train_dataset.categorical_features
    )
    mem_ratio = (ratio < 1/3).mean()
    tau = np.linspace(0.01, 0.99, 99)
    mem_auc = []
    for t in tau:
        mem_auc.append((ratio < t).mean())
    mem_auc = np.mean(mem_auc)
    
    print(f"MemRatio: {mem_ratio:.3f}")
    wandb.log({"MemRatio": mem_ratio})
    print(f"MemAUC: {mem_auc:.3f}")
    wandb.log({"MemAUC": mem_auc})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()
#%%