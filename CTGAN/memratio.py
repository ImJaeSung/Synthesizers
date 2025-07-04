# %%
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import argparse
import importlib

import numpy as np
import torch

from modules.utils import set_random_seed, memorization_ratio
from modules.model import validate_discrete_columns, apply_activate
from modules.data_sampler import DataSampler

from synthetic_eval import evaluation

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
# entity = ""

run = wandb.init(
    project=project, # put your WANDB project name
    # entity=entity, # put your WANDB username
    tags=["memorization"], # put tags of this python project
)
#%%
def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    parser.add_argument("--model", type=str, default="CTGAN")
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        [Tabular dataset options]
                        imbalanced: whitewine, bankruptcy, BAF
                        balanced: breast, banknote, default
                        etc: kings, abalone, anuran, shoppers, magic, creditcard
                        """)
    parser.add_argument('--epochs', default=1000, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=1000, type=int,
                        help='batch size')
    parser.add_argument("--latent_dim", default=128, type=int,
                        help="the latent dimension size")
    parser.add_argument(
        "--generator_dim",
        type=int,
        default=256,
        help="Dimension of each generator layer. "
        "Comma separated integers with no whitespaces.",
    )
    parser.add_argument(
        "--discriminator_dim",
        type=int,
        default=256,
        help="Dimension of each discriminator layer. "
        "Comma separated integers with no whitespaces.",
    )

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False))  # default configuration
    """model load"""
    model_name = f"{config['model']}_{config['latent_dim']}_{config['batch_size']}_{config['epochs']}_{config['generator_dim']}_{config['discriminator_dim']}_{config['dataset']}"
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
        config, train=True)
    test_dataset = CustomDataset(
        config, train=False)

    assert validate_discrete_columns(
        train_dataset.raw_data, 
        train_dataset.EncodedInfo.categorical_features
    ) is None
    """training-by-sampling"""
    data_sampler = DataSampler(
        train_dataset.data, 
        train_dataset.EncodedInfo.transformer.output_info_list, 
        config["log_frequency"]
    )
    config["data_dim"] = train_dataset.EncodedInfo.transformer.output_dimensions
    # %%
    """model"""
    model_module = importlib.import_module("modules.model")
    importlib.reload(model_module)
    
    generator_dim = [config["generator_dim"] for _ in range(2)]
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
    """evaluation"""
    results = evaluation.evaluate(
        syndata, 
        train_dataset.raw_data,
        test_dataset.raw_data, 
        train_dataset.ClfTarget, 
        train_dataset.continuous_features, 
        train_dataset.categorical_features, 
        device)

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
        train_dataset.raw_data,  
        syndata, 
        train_dataset.continuous_features, 
        train_dataset.categorical_features
    )
    test_ratio = memorization_ratio(
        train_dataset.raw_data, 
        test_dataset.raw_data, 
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