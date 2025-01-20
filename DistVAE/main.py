#%%
"""
Reference:
[1] https://github.com/an-seunghwan/DistVAE/blob/main/main.py
"""
#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import ast
#%%
import importlib
import json
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.simulation import set_random_seed
#%%
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

run = wandb.init(
    project="DistVAE.ver3", # put your WANDB project name
    # entity="", # put your WANDB username
    tags=['Train'], # put tags of this python project
)
#%%
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

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
    
    parser.add_argument(
        '--seed', type=int, default=0, help='seed for repeatable results'
    )
    parser.add_argument(
        '--dataset',
        type=str, 
        default='abalone', 
        help='Dataset options: abalone, adult, banknote, breast, cabs, concrete, covtype, credit, kings, letter, loan, redwine, spam, whitewine, yeast'
    )
    parser.add_argument(
        '--test_size', type=float, default=0.2, help='split train and test'
    )
    
    parser.add_argument(
        "--latent_dim", default=100, type=int, help="the latent dimension size"
    )
    parser.add_argument(
        "--step", default=0.1, type=float, help="interval size of quantile levels"
    )
    
    parser.add_argument('--epochs', default=100, type=int, help='the number of epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    
    parser.add_argument(
        '--threshold', default=1e-5, type=float, help='threshold for clipping alpha_tilde'
    )
    parser.add_argument(
        '--beta', default=0.5, type=float, help='scale parameter of asymmetric Laplace distribution'
    )

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
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.{}_dataset'.format(config["dataset"]))
    CustomDataset = dataset_module.CustomDataset
    
    train_dataset = CustomDataset(
        config,
        train=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    EncodedInfo_list = train_dataset.EncodedInfo_list
    CRPS_dim = sum([x.dim for x in EncodedInfo_list if x.activation_fn == 'CRPS'])
    softmax_dim = sum([x.dim for x in EncodedInfo_list if x.activation_fn == 'softmax'])
    config["CRPS_dim"] = CRPS_dim
    config["softmax_dim"] = softmax_dim
    #%%
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)

    model = model_module.VAE(config, device).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    print(model.train())
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params/1000}k")
    wandb.log({"Number of Parameters": num_params/1000})
    #%%
    train_module = importlib.import_module('modules.train')
    for epoch in range(config["epochs"]):
        logs = train_module.train_VAE(
            EncodedInfo_list, 
            train_dataloader, 
            model, 
            config, 
            optimizer,
            device
        )
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join(
            [', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()]
        )
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    """model save"""
    model_dir = f"./assets/models/{config['dataset']}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"DistVAE_beta{config['beta']}_{config['dataset']}_{config['seed']}"

    if config["SmallerGen"]:
        model_name = "small_" + model_name
    
    torch.save(model.state_dict(), f"./{model_dir}/{model_name}.pth")
    
    with open(f"./{model_dir}/config_{config['seed']}.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)    
    
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config) 
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./modules/model.py')
    artifact.add_file('./modules/train.py')
    wandb.log_artifact(artifact)
    #%%    
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%