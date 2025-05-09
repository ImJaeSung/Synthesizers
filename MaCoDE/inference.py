#%%
import os
import torch
import argparse
import importlib
import numpy as np

from modules import utility
from modules.utility import set_random_seed

import warnings
warnings.filterwarnings('ignore')
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "2stage_baseline"
# entity = "uos_stat"

run = wandb.init(
    project=project, # put your WANDB project name
    # entity=entity, # put your WANDB username
    tags=["inference"], # put tags of this python project
)
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    parser.add_argument('--model', type=str, default='macode')
    
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
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate") 
    
    parser.add_argument("--dim_transformer", default=128, type=int,
                        help="the model dimension size")  
    parser.add_argument("--num_transformer_heads", default=8, type=int,
                        help="the number of heads in transformer")
    parser.add_argument("--num_transformer_layer", default=2, type=int,
                        help="the number of layer in transformer") 
    # parser.add_argument("--transformer_dropout", default=0., type=float,
    #                     help="the rate of drop out in transformer") 
    
    parser.add_argument("--bins", default=100, type=int,
                        help="the number of bins used for quantization")
    
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split")     
    parser.add_argument('--epochs', default=1000, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='batch size')
    
    parser.add_argument('--loss', type=str, default='multiclass', 
                        help="multiclass, RPS")
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=0., type=float,
                        help='parameter of AdamW')
    parser.add_argument("--tau", default=1, type=float,
                        help="user defined temperature for privacy controlling") 
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    model_name = "_".join([str(y) for x, y in config.items() if x != "ver" and x != "tau"]) 
    if config["missing_type"] != "None":
        model_name = f"{config['missing_type']}_{config['missing_rate']}_" + model_name
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.mps.is_available() else
        'cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)
    #%%
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset

    """dataset"""
    train_dataset = CustomDataset(
        config, train=True)
    test_dataset = CustomDataset(
        config, scalers=train_dataset.scalers, train=False)
    #%%
    """model"""
    model_module = importlib.import_module("modules.model")
    importlib.reload(model_module)
    model = getattr(model_module, "MaCoDE")(
        config, train_dataset.EncodedInfo, device
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_dir + "/" + model_name,
            map_location=device,
        )
    )
    model.eval()
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000000:.3f}M")
    wandb.log({"Number of Parameters": num_params / 1000000})
    #%%
    if config["missing_type"] == "None":
        n = len(train_dataset.raw_data)
        syndata = model.generate_synthetic_data(n, train_dataset, config["tau"])
    else:
        syndata = model.impute(train_dataset, config["tau"])
    #%%
    from synthetic_eval import evaluation
    results = evaluation.evaluate(
        syndata, train_dataset.raw_data, test_dataset.raw_data, 
        train_dataset.ClfTarget, train_dataset.continuous_features, train_dataset.categorical_features, device
    )
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    
    if config["missing_type"] == "None":
        print("\nMarginal Distribution Visualization...")
        figs = utility.marginal_plot(train_dataset.raw_data, syndata, config)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()
#%%