#%%
import os
import sys

import importlib
import argparse
import ast

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.utils import set_random_seed
#%%
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project="GOGGLE" # put your WANDB project name
# entity = "shoon06" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)

#%%
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v

def get_args(debug=False):
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument('--seed', default=0, type=int, 
                        help="seed for repreatable results")
    parser.add_argument('--dataset', type=str, default='breast', 
                        help="""
                        Tabular dataset options: 
                        breast, banknote, default, whitewine, bankruptcy, BAF
                        """)
        
    parser.add_argument('--test_size', default=0.2, type=float, 
                        help="Proportion of the dataset to include in the test split")
    parser.add_argument('--epochs', default=1000, type=int, 
                        help="Number of training epochs")
    parser.add_argument('--batch_size', default=64, type=int, 
                        help="Batch size for training")
    parser.add_argument('--lr', default=0.001, type=float, 
                        help="Learning rate")
    parser.add_argument('--weight_decay', default=1e-3, type=float, 
                        help="Weight decay (L2 regularization)")
    
    parser.add_argument('--encoder_dim', default=64, type=int, 
                        help="Dimension of the encoder")
    parser.add_argument('--encoder_l', default=2, type=int, 
                        help="Number of layers in the encoder")
    parser.add_argument('--het_encoding', default=True, type=bool, 
                        help="Enable heterogeneous encoding")

    parser.add_argument('--decoder_dim', default=64, type=int, 
                        help="Dimension of the decoder")
    parser.add_argument('--decoder_l', default=2, type=int, 
                        help="Number of layers in the decoder")
    parser.add_argument('--threshold', default=0.1, type=float, 
                        help="Threshold value for graph learning")
    parser.add_argument('--decoder_arch', default="gcn", type=str, 
                        help="Decoder architecture (e.g., 'gcn')")
    
    parser.add_argument('--graph_prior', default=None, type=str,
                        help="Graph prior information")
    parser.add_argument('--prior_mask', default=None, type=str, 
                        help="Graph prior mask")
            
    parser.add_argument('--alpha', default=0.1, type=float, 
                        help='Alpha value for GoggleLoss (KL divergence)')
    parser.add_argument('--beta', default=0.1, type=float, 
                        help='Beta value for GoggleLoss (Graph sparsity)')
    
    parser.add_argument('--iter_opt', default=True, type=bool, 
                        help="Enable iterative optimization")

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()    
#%%
def main():
    #%%
    config = vars(get_args(debug=False))  # default configuration
    # config["cuda"] = torch.cuda.is_available()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    print('Current device is', device)
    set_random_seed(config["seed"])
    wandb.config.update(config)
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset

    train_dataset = CustomDataset(
        config, 
        train=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator(),
    )
    next(iter(train_dataloader))
    config["input_dim"] = train_dataset.EncodedInfo.num_features
#%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.Goggle(config, device)
    model.to(device)
    model.train()
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params/1000}k")
    wandb.log({"Number of Parameters": num_params/1000})
    #%%
    """train"""
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    #%%
    if config['iter_opt']:
        gl_params = ["learned_graph.graph"]
        graph_learner_params = list(
            filter(lambda kv: kv[0] in gl_params, model.named_parameters())
        )
        graph_autoencoder_params = list(
            filter(lambda kv: kv[0] not in gl_params, model.named_parameters())
        )

        optimizer_gl = optim.Adam(
            [param[1] for param in graph_learner_params],
            lr=config["lr"],
            weight_decay=0,
        )
        optimizer_ga = optim.Adam(
            [param[1] for param in graph_autoencoder_params],
            lr=config["lr"],
            weight_decay=config['weight_decay']
        )
        train_module.train_function_iter(
            train_dataloader,
            model,
            optimizer_ga,
            optimizer_gl,
            config,
            device
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config['weight_decay']
        )    
    
        train_module.train_function(
            train_dataloader, 
            model, 
            optimizer,
            config, 
            device
        )
    #%%
    """model save"""
    base_name = f"{config['dataset']}_{config['lr']}_{config['batch_size']}_{config['alpha']}_{config['beta']}"
    model_dir = f"./assets/models/{base_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"GOGGLE_{base_name}_{config['seed']}"

    torch.save(model.state_dict(), f"./{model_dir}/{model_name}.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config)
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./modules/model.py')
    artifact.add_file('./modules/train.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%    
if __name__ == '__main__':
    main()

