# %%
import os
import io
import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np

import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.simulation import set_random_seed

from modules.model import validate_discrete_columns, apply_activate
from modules.data_sampler import DataSampler

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

project = "distvae_journal_baseline1" # put your WANDB project name
entity = "anseunghwan" # put your WANDB username

run = wandb.init(
    project=project, 
    entity=entity, 
    tags=["viz"], # put tags of this python project
)
# %%
def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--ver', type=int, default=1, 
                        help='model version number')
    parser.add_argument("--model", type=str, default="CTGAN")
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        Tabular dataset options: 
                        banknote, whitewine, breast, bankruptcy, musk, madelon
                        """)
    
    parser.add_argument("--latent_dim", default=512, type=int,
                        help="the latent dimension size")

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

# %%
def main():
    # %%
    config = vars(get_args(debug=True))  # default configuration

    """model load"""
    model_name = f"{config['model']}_{config['latent_dim']}_{config['dataset']}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)
    # %%
    """dataset"""
    dataset_module = importlib.import_module(f"datasets.preprocess")
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    
    train_dataset = CustomDataset(
        config, train=True
    )

    assert validate_discrete_columns(
        train_dataset.raw_data, 
        train_dataset.EncodedInfo.categorical_features
    ) is None
    # %%
    """training-by-sampling"""
    data_sampler = DataSampler(
        train_dataset.data, train_dataset.EncodedInfo.transformer.output_info_list, config["log_frequency"]
    )
    config["data_dim"] = train_dataset.EncodedInfo.transformer.output_dimensions
    # %%
    """model"""
    model_module = importlib.import_module("modules.model")
    importlib.reload(model_module)
    
    generator_dim = [int(x) for x in config["generator_dim"].split(",")]
    model = getattr(model_module, "Generator")(
        config["latent_dim"] + data_sampler.dim_cond_vec(),
        generator_dim,
        config["data_dim"],
    ).to(device)

    if config["cuda"]:
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name
            )
        )
    else:
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name, 
                map_location=torch.device('cpu'),
            )
        )
    model.eval()
    #%%
    """Number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_generator_params = count_parameters(model)
    print(f"Number of Generator Parameters: {num_generator_params / 1000:.2f}K")
    wandb.log({"Number of Generator Parameters": num_generator_params / 1000})
    # %%
    """synthetic dataset generation """
    n = len(train_dataset.raw_data)
    syndata = model.generate_synthetic_data(n, train_dataset, data_sampler, config, device)
    #%%
    fig, ax = plt.subplots(
        2, len(train_dataset.continuous_features) // 2 + 1, 
        figsize=(len(train_dataset.continuous_features) + 2, 5),
        sharey=True
    )
    for i, col in enumerate(train_dataset.continuous_features):
        true = train_dataset.raw_data[col].sort_values()
        true_CDF = np.linspace(0, 1, len(true))
        syn = syndata[col].sort_values()
        syn_CDF = np.linspace(0, 1, len(syn))

        ax.flatten()[i].plot(true, true_CDF, label="train", linewidth=3)
        ax.flatten()[i].plot(syn, syn_CDF, label="synthetic", linestyle="--", linewidth=3)
        ax.flatten()[i].set_xlabel(col, fontsize=14)
        ax.flatten()[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        # ax.flatten()[i].axhline(y=1.0, color='k', linestyle='--', linewidth=0.5)
        # ax.flatten()[i].axhline(y=0.0, color='k', linestyle='--', linewidth=0.5)
    ax.flatten()[0].set_ylabel('CDF', fontsize=14)
    ax.flatten()[-1].axis("off")
    plt.legend()
    plt.tight_layout()
    basedir = "/Users/anseunghwan/Documents/GitHub/distvae_journal/assets/figs/overleaf"
    plt.savefig(basedir + f"/CDF/{config['model']}_{config['dataset']}_CDF.png", bbox_inches='tight')
    plt.show()
    plt.close()
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
# %%
if __name__ == "__main__":
    main()
# %%
