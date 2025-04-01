# %%
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import argparse
import importlib

from modules.utils import set_random_seed, warmup_lr, infiniteloop

import torch
from torch.utils.data import DataLoader
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

project = "2stage_baseline_debug" # put your WANDB project name
# entity = "" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["inference"], # put tags of this python project
)
# %%
def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    parser.add_argument("--model", type=str, default="CoDi")
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        [Tabular dataset options]
                        imbalanced: whitewine, bankruptcy, BAF
                        balanced: breast, banknote, default
                        etc: kings, abalone, anuran, shoppers, magic, creditcard
                        """)
    parser.add_argument('--epochs', default=20000, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=2100, type=int,
                        help='batch size')
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

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

# %%
def main():
    # %%
    config = vars(get_args(debug=False))  # default configuration
    #%%
    """model load"""
    # continuous
    model_name1 = f"continuous_{config['model']}_{config['embed_dim_Cont']}_{config['embed_dim_Disc']}_{config['epochs']}_{config['Cont_encoder_dim_']}_{config['Disc_encoder_dim_']}_{config['Cont_lambda']}_{config['Disc_lambda']}_{config['dataset']}"
    artifact1 = wandb.use_artifact(
        f"{project}/{model_name1}:v{config['ver']}",
        type='model')
    for key, item in artifact1.metadata.items():
        config[key] = item
    model_dir1 = artifact1.download()
    model_name1 = [x for x in os.listdir(model_dir1) 
                   if x.startswith('continuous') and x.endswith(f"{config['seed']}.pth")][0]
    #%%
    # discrete
    model_name2 = f"discrete_{config['model']}_{config['embed_dim_Cont']}_{config['embed_dim_Disc']}_{config['epochs']}_{config['Cont_encoder_dim_']}_{config['Disc_encoder_dim_']}_{config['Cont_lambda']}_{config['Disc_lambda']}_{config['dataset']}"
    artifact2 = wandb.use_artifact(
        f"{project}/{model_name2}:v{config['ver']}",
        type='model')
    for key, item in artifact2.metadata.items():
        config[key] = item
    model_dir2 = artifact2.download()
    
    model_name2 = [x for x in os.listdir(model_dir2) 
                   if x.startswith('discrete') and x.endswith(f"{config['seed']}.pth")][0]
    
    #%%
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)
    # %%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset

    train_dataset = CustomDataset(config, train=True)
    test_dataset = CustomDataset(config, train=False)
    
    C = train_dataset.num_continuous_features
    train_dataset_Cont = train_dataset.data[:, :C]
    train_dataset_Disc = train_dataset.data[:, C:]
    # %%
    """model"""
    model_module = importlib.import_module('modules.tabularUnet')
    importlib.reload(model_module)

    config["Cont_encoder_dim"] = [config["Cont_encoder_dim_"] * (2 ** i) for i in range(3)]
    tabularUnet_Cont = model_module.tabularUnet(
        config, train_dataset_Cont, train_dataset_Disc, Continuous=True).to(device)

    
    if config["cuda"]:
        tabularUnet_Cont.load_state_dict(
            torch.load(
                model_dir1 + '/' + model_name1
            )
        )
    else:
        tabularUnet_Cont.load_state_dict(
            torch.load(
                model_dir1 + '/' + model_name1, 
                map_location=torch.device('cpu'),
            )
        )
    tabularUnet_Cont.eval()
    #%%
    config["Disc_encoder_dim"] = [config["Disc_encoder_dim_"] * (2 ** i) for i in range(3)]
    tabularUnet_Disc = model_module.tabularUnet(
        config, train_dataset_Cont, train_dataset_Disc, Continuous=False).to(device)
    if config["cuda"]:
        tabularUnet_Disc.load_state_dict(
            torch.load(
                model_dir2 + '/' + model_name2
            )
        )
    else:
        tabularUnet_Disc.load_state_dict(
            torch.load(
                model_dir2 + '/' + model_name2, 
                map_location=torch.device('cpu'),
            )
        )
    tabularUnet_Disc.eval()
    #%%
    trainer_module = importlib.import_module('modules.diffusion')
    importlib.reload(trainer_module)
    Sampler_Cont = trainer_module.GaussianDiffusionSampler(tabularUnet_Cont, config).to(device)
    Trainer_Disc = trainer_module.MultinomialDiffusion(train_dataset, tabularUnet_Disc, config).to(device)
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_Cont = count_parameters(tabularUnet_Cont)
    num_params_Disc = count_parameters(tabularUnet_Disc)
    
    print(f"Number of Parameters of continuous Unet: {num_params_Cont/1000}k")
    print(f"Number of Parameters of discrete Unet: {num_params_Disc/1000}k")
    
    wandb.log({"Number of Parameters of continuous Unet": num_params_Cont/1000})
    wandb.log({"Number of Parameters of discrete Unet": num_params_Disc/1000})
    # %%
    """synthetic dataset generation """
    importlib.reload(model_module)
    syndata = model_module.generate_synthetic_data(
        train_dataset, Sampler_Cont, Trainer_Disc, config, device)
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
    
    # print("Marginal Distribution...")
    # figs = utility.marginal_plot(train_dataset.raw_data, syndata, config, model_name)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
# %%
if __name__ == "__main__":
    main()
# %%
