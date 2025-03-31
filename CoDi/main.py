# %%
import os
import sys
import importlib
import argparse
import ast
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# %%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from modules.utils import set_random_seed, warmup_lr, infiniteloop

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
# %%
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding="utf-8")
    import wandb

# project = "2stage_baseline" # put your WANDB project name
project = "debug" # put your WANDB project name
# entity = "" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)
# %%
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v


def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument("--model", type=str, default="CoDi")
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        [Tabular dataset options]
                        imbalanced: whitewine, bankruptcy, BAF
                        balanced: breast, banknote, default
                        etc: kings, abalone, anuran, shoppers, magic, creditcard
                        """)
    
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split")     
    parser.add_argument('--epochs', default=20000, type=int,
                        help='total training steps')
    parser.add_argument('--batch_size', default=2100, type=int,
                        help='batch size')
    parser.add_argument('--activation_fn', default="relu", type=str,
                        help='activation function for continuous space')
        
    parser.add_argument('--embed_dim_Cont', default=16, type=int,
                        help='embedding dimension in Unet')
    parser.add_argument('--embed_dim_Disc', default=64, type=int,
                        help='embedding dimension in Unet')
    
    parser.add_argument(
        "--Cont_encoder_dim_",
        type=int,
        default=64,
        help="Dimension of encoder layer. (x, 2x 3x)"
    )
    parser.add_argument(
        "--Disc_encoder_dim_",
        type=int,
        default=64,
        help="Dimension of decoder layer. (x, 2x 3x)"
    )
    
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=50,
        help="total diffusion steps T.",
    )
    
    parser.add_argument(
        "--beta_init",
        type=float,
        default=1e-5,
        help="start beta value.",
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="end beta value",
    )
    
    parser.add_argument(
        "--Cont_lr",
        type=float,
        default=2e-3,
        help="Learning rate for the continuous diffusion.",
    )
    parser.add_argument(
        "--Disc_lr",
        type=float,
        default=2e-3,
        help="Learning rate for the discrete diffusion.",
    )
    
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.,
        help="gradient norm cliping",
    )

    parser.add_argument(
        '--mean_type', type=str, default='epsilon', 
        help="""
        predict variable
        [options] xprev, xstart, epsilon
        """
    )
    
    parser.add_argument(
        '--var_type', type=str, default='fixedsmall', 
        help="""
        variance type
        [options] fixedlarge, fixedsmall
        """
    )
    
    parser.add_argument(
        "--ns_method",
        type=int,
        default=0,
        help="negative condition method.",
    )
    
    parser.add_argument(
        "--Cont_lambda",
        type=float,
        default=0.2,
        help="lambda in objective function for continuous variable"
    )
    
    parser.add_argument(
        "--Disc_lambda",
        type=float,
        default=0.2,
        help="lambda in objective function for discrete variable"
    )
    
    parser.add_argument('--sampling_step', default=2000, type=int,
                        help='frequency of sampling')
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

# %%
def main():
    # %%
    config = vars(get_args(debug=False))  # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    wandb.config.update(config)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    # %%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset

    train_dataset = CustomDataset(config, train=True)
    C = train_dataset.num_continuous_features

    train_dataset_Cont = train_dataset.data[:, :C]
    train_dataset_Disc = train_dataset.data[:, C:]
    
    train_dataloader_Cont = DataLoader(
        train_dataset_Cont,
        batch_size=config["batch_size"],
        shuffle=True
    )
    train_dataloader_Disc = DataLoader(
        train_dataset_Disc,
        batch_size=config["batch_size"],
        shuffle=True
    )
    train_datalooper_Cont = infiniteloop(train_dataloader_Cont)
    train_datalooper_Disc = infiniteloop(train_dataloader_Disc)
    
    # %%
    """model"""
    model_module = importlib.import_module('modules.tabularUnet')
    importlib.reload(model_module)

    config["Cont_encoder_dim"] = [config["Cont_encoder_dim_"] * (2 ** i) for i in range(3)]
    tabularUnet_Cont = model_module.tabularUnet(
        config, train_dataset_Cont, train_dataset_Disc, Continuous=True).to(device)
    
    config["Disc_encoder_dim"] = [config["Disc_encoder_dim_"] * (2 ** i) for i in range(3)]
    tabularUnet_Disc = model_module.tabularUnet(
        config, train_dataset_Cont, train_dataset_Disc, Continuous=False).to(device)
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_Cont = count_parameters(tabularUnet_Cont)
    num_params_Disc = count_parameters(tabularUnet_Disc)
    
    print(f"Number of Parameters of continuous Unet: {num_params_Cont/1000}k")
    print(f"Number of Parameters of discrete Unet: {num_params_Disc/1000}k")
    
    wandb.log({"Number of Parameters of continuous Unet": num_params_Cont/1000})
    wandb.log({"Number of Parameters of discrete Unet": num_params_Disc/1000})
    #%%
    """train"""
    trainer_module = importlib.import_module('modules.diffusion')
    importlib.reload(trainer_module)
    
    Trainer_Cont = trainer_module.GaussianDiffusionTrainer(tabularUnet_Cont, config).to(device)
    Trainer_Disc = trainer_module.MultinomialDiffusion(train_dataset, tabularUnet_Disc, config).to(device)
    
    optimizer_Cont = optim.Adam(tabularUnet_Cont.parameters(), lr=config["Cont_lr"])
    scheduler_Cont = optim.lr_scheduler.LambdaLR(optimizer_Cont, lr_lambda=warmup_lr)
    optimizer_Disc = optim.Adam(tabularUnet_Disc.parameters(), lr=config["Disc_lr"])
    scheduler_Disc = optim.lr_scheduler.LambdaLR(optimizer_Disc, lr_lambda=warmup_lr)
    
    tabularUnet_Cont.train() 
    tabularUnet_Disc.train()
    
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    train_module.train_function(
        EncodedInfo_list=train_dataset.EncodedInfo_list,
        train_datalooper_Cont=train_datalooper_Cont,
        train_datalooper_Disc=train_datalooper_Disc, 
        Trainer_Cont=Trainer_Cont,
        Trainer_Disc=Trainer_Disc,
        tabularUnet_Cont=tabularUnet_Cont,
        config=config, 
        optimizer_Cont=optimizer_Cont,
        optimizer_Disc=optimizer_Disc,
        scheduler_Cont=scheduler_Cont,
        scheduler_Disc=scheduler_Disc, 
        device=device
    )
    # %%
    """model save"""
    base_name = f"{config['model']}_{config['embed_dim_Cont']}_{config['embed_dim_Disc']}_{config['epochs']}_{config['Cont_encoder_dim_']}_{config['Disc_encoder_dim_']}_{config['Cont_lambda']}_{config['Disc_lambda']}_{config['dataset']}"
    
    # continuous diffusion
    model_dir1 = f"./assets/models/continuous/{base_name}/"
    if not os.path.exists(model_dir1):
        os.makedirs(model_dir1)
    model_name1 = f"continuous_{base_name}_{config['seed']}"
    torch.save(tabularUnet_Cont.state_dict(), f"./{model_dir1}/{model_name1}.pth")
    artifact1 = wandb.Artifact(
        "_".join(model_name1.split("_")[:-1]), 
        type='model',
        metadata=config) 
    artifact1.add_file(f"./{model_dir1}/{model_name1}.pth")
    artifact1.add_file('./main.py')
    artifact1.add_file('./datasets/preprocess.py')
    artifact1.add_file('./modules/train.py')
    artifact1.add_file('./modules/diffusion.py')
    artifact1.add_file('./modules/tabularUnet.py')
    wandb.log_artifact(artifact1)
    #%%
    # discrete diffusion
    model_dir2 = f"./assets/models/discrete/{base_name}/"
    if not os.path.exists(model_dir2):
        os.makedirs(model_dir2)
    model_name2 = f"discrete_{base_name}_{config['seed']}"
    torch.save(tabularUnet_Disc.state_dict(), f"./{model_dir2}/{model_name2}.pth")
    artifact2 = wandb.Artifact(
        "_".join(model_name2.split("_")[:-1]), 
        type='model',
        metadata=config) 
    artifact2.add_file(f"./{model_dir2}/{model_name2}.pth")
    artifact2.add_file('./main.py')
    artifact2.add_file('./datasets/preprocess.py')
    artifact2.add_file('./modules/train.py')
    artifact2.add_file('./modules/diffusion.py')
    artifact2.add_file('./modules/tabularUnet.py')
    wandb.log_artifact(artifact2)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
# %%
if __name__ == "__main__":
    main()