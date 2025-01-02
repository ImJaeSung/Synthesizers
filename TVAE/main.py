# %%
"""
Reference:
[1] https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py
"""
#%%
import os
import importlib
import sys
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from modules.utils import set_random_seed

import warnings
warnings.filterwarnings(action='ignore')
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

project = "TVAE_2stage" # put your WANDB project name
# entity = "" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)
# %%
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='banknote', 
                        help="""
                        [Tabular dataset options]
                        imbalanced: whitewine, bankruptcy, BAF
                        balanced: breast, banknote, default
                        etc: kings, abalone, anuran, shoppers, magic, creditcard
                        """)
    
    parser.add_argument("--latent_dim", default=128, type=int,
                        help="the latent dimension size")
    
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split")     
    parser.add_argument('--epochs', default=300, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=500, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--loss_factor', default=2, type=float,
                        help='weight in ELBO')
    
    parser.add_argument(
        "--sigma_range",
        default=[0.01, 1],
        type=arg_as_list,
        help="range of observational noise",
    )

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

    train_dataset = CustomDataset(
        config, train=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"])
    config["input_dim"] = train_dataset.EncodedInfo.transformer.output_dimensions
    # %%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.TVAE(config, device).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=1e-5
    )
    model.train()
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params/1000:.2f}K")
    # %%
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    
    for epoch in range(config["epochs"]):
        logs = train_module.train(
            train_dataset.EncodedInfo.transformer.output_info_list,
            train_dataset,
            train_dataloader,
            model,
            config,
            optimizer,
            device,
        )

        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += "".join(
            [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
        )
        print(print_input)

        """update log"""
        wandb.log({x: np.mean(y) for x, y in logs.items()})
    # %%
    """model save"""
    base_name = f"{project}_{config['latent_dim']}_{config['epochs']}_{config['batch_size']}_{config['loss_factor']}_{config['dataset']}"
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
    artifact.add_file('./modules/train.py')
    artifact.add_file('./modules/model.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
# %%
if __name__ == "__main__":
    main()
# %%
