#%%
import os
import argparse
import importlib

import torch
from torch.utils.data import DataLoader

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
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument("--model", type=str, default="TabMT")
    
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        Dataset options: 
                        abalone, banknote, breast, concrete, covtype,
                        kings, letter, loan, redwine, whitewine
                        """)
    
    parser.add_argument("--missing_type", default="None", type=str,
                        help="""
                        how to generate missing: None(complete data), MCAR, MAR, MNARL, MNARQ
                        """) 
    parser.add_argument("--dim_transformer", default=128, type=int,
                        help="the model dimension size")  
    parser.add_argument("--num_transformer_heads", default=4, type=int,
                        help="the number of heads in transformer")
    parser.add_argument("--transformer_dropout", default=0., type=float,
                        help="the rate of drop out in transformer") 
    parser.add_argument("--num_transformer_layer", default=2, type=int,
                        help="the number of layer in transformer") 
    
    parser.add_argument("--max_clusters", default=10, type=int,
                        help="the number of bins used quantization")
    
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split")       
    parser.add_argument('--epochs', default=10000, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=1024, type = int,
                        help='batch size')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=0.002, type=float,
                        help='parameter of AdamW')
    
    parser.add_argument("--tau", default=1.0, type=float,
                        help="user defined temperature for privacy controlling") 
    parser.add_argument('--SmallerGen', default=False, type=str2bool,
                        help='Use smaller batch size for synthetic data generation') 
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current cuda device is', device)
    wandb.config.update(config)
    #%%
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset

    """dataset"""
    train_dataset = CustomDataset(
        config,
        kmeans_models=False,
        train=True,)

    train_dataloader = DataLoader(
        train_dataset, 
        config['batch_size'], 
        shuffle=True, 
        drop_last=False)
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.TabMT(config, train_dataset.EncodedInfo, device).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'], 
        eta_min=0)
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000:.2f}k")
    wandb.log({"Number of Parameters": num_params / 1000})
    #%%
    """train"""
    if config["missing_type"] is None:
        train_module = importlib.import_module('modules.train')
    else:
        train_module = importlib.import_module('modules.missing_train')
    importlib.reload(train_module)
    train_module.train_function(
        model,
        config,
        optimizer, 
        scheduler, 
        train_dataloader,
        device)
    #%%
    """model save"""
    base_name = f"{config['model']}_{config['missing_type']}_{config['dim_transformer']}_{config['num_transformer_heads']}_{config['max_clusters']}_{config['num_transformer_layer']}_{config['epochs']}_{config['tau']}_{config['dataset']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"
    torch.save(model.state_dict(), f"./{model_dir}/{model_name}.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config) 
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./datasets/preprocess.py')
    artifact.add_file('./modules/model.py')
    if config["missing_type"] == "None":
        artifact.add_file('./modules/train.py')
    else:
        artifact.add_file('./modules/missing_train.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%%
if __name__ == '__main__':
    main()