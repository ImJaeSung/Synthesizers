#%%
import os
import io
from PIL import Image
import torch
import argparse
import importlib

import numpy as np

from modules.utils import get_model

import modules
from evaluation.evaluation import evaluate
from evaluation.utility import set_random_seed

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

project = "ddpm" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["inference"], # put tags of this python project
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
    
    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    parser.add_argument('--dataset', type=str, default='kings', 
                        help="""
                        Dataset options: 
                        abalone, adult, banknote, breast, concrete, 
                        kings, loan, covertype, redwine, whitewine
                        """)
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    #%%
    """model load"""
    base_name = f"TabDDPM_{config['dataset']}"
    model_name = f"{base_name}"
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
    #%%
    dataset_module = importlib.import_module('dataset.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    
    """dataset"""
    train_dataset = CustomDataset(
        config,
        train=True
    )
    test_dataset = CustomDataset(
        config,
        train=False,
        cont_scalers=train_dataset.cont_scalers,
        disc_scalers=train_dataset.disc_scalers
    )
    #%%\
    K = np.array(train_dataset.EncodedInfo.num_categories)
    num_numerical_features = train_dataset.EncodedInfo.num_continuous_features
    d_in = np.sum(K) + num_numerical_features
    
    config['is_y_cond'] = True
    config['num_classes'] = train_dataset.num_classes
    config['d_in'] = d_in
    #%%
    """model"""
    model = get_model(config)
    model.to(device)
    #%%
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    diffusion = model_module.GaussianMultinomialDiffusion(
        EncodedInfo=train_dataset.EncodedInfo,
        denoise_fn=model,
        config=config,
        device=device
    ).to(device)

    if config["cuda"]:
        diffusion.load_state_dict(
            torch.load(
                model_dir + "/" + model_name
            )
        )
    else:
        diffusion.load_state_dict(
            torch.load(
                model_dir + "/" + model_name,
                map_location=torch.device("cpu"),
            )
        )
    diffusion.eval()
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000000:.1f}M")
    wandb.log({"Number of Parameters": num_params / 1000000})
    #%%
    n = len(train_dataset.raw_data)
    syndata = diffusion.generate_synthetic_data(
        num_samples=n, 
        train_dataset=train_dataset,
        ddim=False)
    #%%
    results = evaluate(syndata, train_dataset, test_dataset, config)
    results = results._asdict()

    for x, y in results.items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()
#%%