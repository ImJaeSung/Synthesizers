#%%
import os
import torch
import argparse
import importlib
import matplotlib.pyplot as plt

from modules.utils import set_random_seed
from modules.utils import generalization_score, DCR_values

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
    parser.add_argument("--max_clusters", default=20, type=int,
                        help="the number of bins used quantization")
    parser.add_argument('--batch_size', default=512, type = int,
                        help='batch size')
    parser.add_argument('--epochs', default=20000, type=int,
                        help='the number of epochs')
    
    parser.add_argument("--dim_transformer", default=512, type=int,
                        help="the model dimension size")  
    parser.add_argument("--num_transformer_heads", default=8, type=int,
                        help="the number of heads in transformer")
    parser.add_argument("--transformer_dropout", default=0., type=float,
                        help="the rate of drop out in transformer") 
    parser.add_argument("--num_transformer_layer", default=8, type=int,
                        help="the number of layer in transformer") 

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
    
    """model load"""
    model_name = f"{config['model']}_{config['missing_type']}_{config['dim_transformer']}_{config['num_transformer_heads']}_{config['max_clusters']}_{config['num_transformer_layer']}_{config['batch_size']}_{config['epochs']}_{config['tau']}_{config['dataset']}"

    if config["SmallerGen"]:
        model_name = "small_" + model_name

    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model'
    )
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()

    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    wandb.config.update(config)
    #%%
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    """dataset"""
    train_dataset = CustomDataset(
        config,
        kmeans_models=None,
        train=True
    )
    test_dataset = CustomDataset(
        config,
        kmeans_models=train_dataset.kmeans_models,
        train=False)
    #%%
    """model"""
    model_module = importlib.import_module("modules.model")
    importlib.reload(model_module)
    model = model_module.TabMT(config, train_dataset.EncodedInfo, device).to(device)
    
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
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000000:.3f}M")
    wandb.log({"Number of Parameters": num_params / 1000000})
    #%%
    """Synthetic data generation"""
    n = len(train_dataset.raw_data)
    syndata = model.generate_synthetic_data(n, train_dataset)
    #%%
    syn_cos = generalization_score(
        train_dataset.raw_data, 
        syndata, 
        train_dataset.continuous_features,
        train_dataset.categorical_features
    )
    test_cos = generalization_score(
        train_dataset.raw_data, 
        test_dataset.raw_data, 
        train_dataset.continuous_features, 
        train_dataset.categorical_features
    )

    threshold = 0.95
    syn_glscore = 1 - (syn_cos > threshold).mean()
    test_glscore = 1 - (test_cos > threshold).mean()

    print(f"Syn_GLScore: {syn_glscore:.5f}")
    wandb.log({"Syn_GLScore": syn_glscore})
    print(f"Test_GLScore: {test_glscore:.5f}")
    wandb.log({"Test_GLScore": test_glscore})
    #%%
    directory = f"./assets/fig_glscore/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    import numpy as np
    import seaborn as sns
    # sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.histplot(
        x=test_cos,
        stat='density',
        label='test',
        bins=int(np.sqrt(len(test_cos))),
        edgecolor='black',
        ax=ax
    )

    sns.histplot(
        x=syn_cos,
        stat='density',
        label='synthetic',
        bins=int(np.sqrt(len(syn_cos))),
        edgecolor='black',
        ax=ax
    )

    ax.set_xlabel('GLScore', fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{directory}/hist_glscore_{config['dataset']}_{config['model']}.png")
    plt.show()
    plt.close()
    #%%
    syn_dcr = DCR_values(
        train_dataset.raw_data, 
        syndata, 
        train_dataset.
        continuous_features, 
        train_dataset.categorical_features
    )
    test_dcr = DCR_values(
        train_dataset.raw_data, 
        test_dataset.raw_data, 
        train_dataset.continuous_features, 
        train_dataset.categorical_features
    )
    #%%
    directory = f"./assets/fig_dcr/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    import numpy as np
    import seaborn as sns
    # sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.histplot(
        x=test_dcr,
        stat='density',
        label='test',
        bins=int(np.sqrt(len(test_dcr))),
        edgecolor='black',
        ax=ax
    )

    sns.histplot(
        x=syn_dcr,
        stat='density',
        label='synthetic',
        bins=int(np.sqrt(len(syn_dcr))),
        edgecolor='black',
        ax=ax
    )

    ax.set_xlabel('DCR', fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{directory}/hist_dcr_{config['dataset']}_{config['model']}.png")
    plt.show()
    plt.close()
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()
#%%