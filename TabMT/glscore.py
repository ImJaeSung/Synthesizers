#%%
import os
import torch
import argparse
import importlib
import matplotlib.pyplot as plt

from modules.utility import set_random_seed
from modules.utility import generalization_score, DCR_values

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

project = "DSGM0"
entity = "anseunghwan"

run = wandb.init(
    project=project, # put your WANDB project name
    entity=entity, # put your WANDB username
    tags=["memorization"], # put tags of this python project
)
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        Dataset options: 
                        abalone, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, whitewine
                        """)
    
    parser.add_argument("--dim_transformer", default=512, type=int,
                        help="the model dimension size")  
    parser.add_argument("--num_transformer_heads", default=8, type=int,
                        help="the number of heads in transformer")
    parser.add_argument("--num_transformer_layer", default=1, type=int,
                        help="the number of layer in transformer") 
    # parser.add_argument("--transformer_dropout", default=0., type=float,
    #                     help="the rate of drop out in transformer") 
    
    parser.add_argument("--M", default=100, type=int,
                        help="the cardinality of latent variable")
    parser.add_argument("--cutpoint", default=4, type=int,
                        help="the start(or end) value of the latent variable set")
    
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split")     
    parser.add_argument('--epochs', default=1000, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    # parser.add_argument('--weight_decay', default=0., type=float,
    #                     help='parameter of AdamW')
        
    parser.add_argument('--beta', default=0.1, type=float,
                        help='decoder variance')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    model_name = "_".join([str(y) for x, y in config.items() if x != 'seed' and x != 'beta' and x != 'gpu_id']) 
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['seed']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else 'cpu')
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
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    model_module = importlib.import_module("modules.model")
    importlib.reload(model_module)
    model = getattr(model_module, "Model")(
        torch.from_numpy(train_dataset.thetas),
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
    n = len(train_dataset.raw_data)
    syndata = model.generate_synthetic_data(n, train_dataset, config['beta'])
    #%%
    syn_cos = generalization_score(
        train_dataset.raw_data, syndata, train_dataset.continuous_features, train_dataset.categorical_features
    )
    test_cos = generalization_score(
        train_dataset.raw_data, test_dataset.raw_data, train_dataset.continuous_features, train_dataset.categorical_features
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
    plt.savefig(f"{directory}/hist_glscore_{config['dataset']}_ours.png")
    plt.show()
    plt.close()
    #%%
    syn_dcr = DCR_values(
        train_dataset.raw_data, syndata, train_dataset.continuous_features, train_dataset.categorical_features
    )
    test_dcr = DCR_values(
        train_dataset.raw_data, test_dataset.raw_data, train_dataset.continuous_features, train_dataset.categorical_features
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
    plt.savefig(f"{directory}/hist_dcr_{config['dataset']}_ours.png")
    plt.show()
    plt.close()
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()
#%%