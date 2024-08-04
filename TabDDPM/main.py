# %%
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import importlib
import numpy as np

import argparse
import ast

import torch
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import get_model
from evaluation.simulation import set_random_seed
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

project = "TabDDPM" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

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

    parser.add_argument("--seed", type=int, default=0, 
                        help="seed for repeatable results")
    parser.add_argument('--dataset', type=str, default='kings', 
                        help="""
                        Dataset options: 
                        abalone, adult, banknote, breast, concrete, 
                        kings, loan, covertype, redwine, whitewine
                        """)
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split")
      
    parser.add_argument("--steps", type=int, default=10000, 
                        help="Number of steps")
    parser.add_argument("--lr", type=float, default=0.002, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                        help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=1024, 
                        help="Batch size")
    
    parser.add_argument("--model_type", type=str, default='mlp', 
                        help="Type of model")
    parser.add_argument("--model_params", type=str, default=None, 
                        help="Parameters of the model")
    parser.add_argument("--num_timesteps", type=int, default=1000, 
                        help="Number of timesteps")
    parser.add_argument("--gaussian_loss_type", type=str, default='mse', 
                        help="Type of Gaussian loss")
    parser.add_argument("--scheduler", type=str, default='cosine', 
                        help="Scheduler type")
    parser.add_argument("--gaussian_parametrization", type=str, default='eps', 
                        help="Gaussian parametrization")
    parser.add_argument("--multinomial_loss_type", type=str,
                        default='vb_stochastic', help="Multinomial loss type")
    parser.add_argument("--parametrization", type=str, default='x0', 
                        help="Parametrization")

    parser.add_argument("--embedding_dim", type=list, default=[128, 128], 
                        help="embedding dimension of TabDDPM")
    
    parser.add_argument("--dropout", type=list, default=0.0, 
                        help="dropout of TabDDPM")

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

# %%
def main():
    # %%
    config = vars(get_args(debug=False))  # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    wandb.config.update(config)
    print(device)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    # %%
    """dataset"""
    dataset_module = importlib.import_module('dataset.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset


    train_dataset = CustomDataset(
        config,
        train=True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size'])
    
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
    )
    diffusion.to(device)
    diffusion.train()
    #%%
    """Train"""
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    trainer = train_module.Trainer(
        diffusion,
        train_dataloader,
        config,
        device=device
    )
    #%%
    trainer.run_loop()
    # %%
    """number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params = count_parameters(diffusion)
    print(f"Number of Parameters: {num_params/ 1000:.2f}k")
    #%%
    """model save"""
    base_name = f"TabDDPM_{config['embedding_dim']}_{config['dataset']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config) 
    artifact.add_file(f"./{model_dir}/{model_name}.pkl")
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