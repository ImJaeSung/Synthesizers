"""
Reference:
[1] https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py
"""
# %%
import os
import sys
import importlib
import argparse
import ast
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# %%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from modules.data_sampler import *
from modules.utils import set_random_seed
from modules.model import validate_discrete_columns
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

project = "2stage_baseline" # put your WANDB project name
# entity = "" # put your WANDB username

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

    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument("--model", type=str, default="CTGAN")
    parser.add_argument('--dataset', type=str, default='whitewine', 
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
    parser.add_argument('--epochs', default=1000, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=1000, type=int,
                        help='batch size')
    
    parser.add_argument(
        "--generator_lr",
        type=float,
        default=2e-4,
        help="Learning rate for the generator.",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=2e-4,
        help="Learning rate for the discriminator.",
    )

    parser.add_argument(
        "--generator_decay",
        type=float,
        default=1e-6,
        help="Weight decay for the generator.",
    )
    parser.add_argument(
        "--discriminator_decay",
        type=float,
        default=1e-6,
        help="Weight decay for the discriminator.",
    )

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

    parser.add_argument(
        "--pac",
        type=int,
        default=10,
        help="Number of samples to group together when applying the discriminator.",
    )
    parser.add_argument(
        "--discriminator_steps",
        type=int,
        default=1,
        help="Number of discriminator updates to do for each generator update."
        "From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper"
        "default is 5. Default used is 1 to match original CTGAN implementation.",
    )
    parser.add_argument(
        "--log_frequency",
        action="store_false",
        help="Whether to use log frequency of categorical levels in conditional sampling."
        "Defaults to True.",
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

    train_dataset = CustomDataset(config, train=True)
    
    assert validate_discrete_columns(
        train_dataset.raw_data, 
        train_dataset.EncodedInfo.categorical_features
    ) is None
    # %%
    """training-by-sampling"""
    data_sampler = DataSampler(
        train_dataset.data, 
        train_dataset.EncodedInfo.transformer.output_info_list, 
        config["log_frequency"]
    )
    config["data_dim"] = train_dataset.EncodedInfo.transformer.output_dimensions
    # %%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)

    generator_dim = [config["generator_dim"] for _ in range(2)]
    discriminator_dim = [config["discriminator_dim"] for _ in range(2)]

    generator = model_module.Generator(
        config["latent_dim"] + data_sampler.dim_cond_vec(),
        generator_dim,
        config["data_dim"],
    ).to(device)

    discriminator = model_module.Discriminator(
        config["data_dim"] + data_sampler.dim_cond_vec(),
        discriminator_dim,
        pac=config["pac"],
    ).to(device)

    optimizerG = torch.optim.Adam(
        generator.parameters(),
        lr=config["generator_lr"],
        betas=(0.5, 0.9),
        weight_decay=config["generator_decay"],
    )

    optimizerD = torch.optim.Adam(
        discriminator.parameters(),
        lr=config["discriminator_lr"],
        betas=(0.5, 0.9),
        weight_decay=config["discriminator_decay"],
    )

    mean = torch.zeros(config["batch_size"], config["latent_dim"], device=device)
    std = mean + 1

    print(generator.train())
    print(discriminator.train())
    # %%
    """number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_generator_params = count_parameters(generator)
    num_discriminator_params = count_parameters(discriminator)
    num_params = count_parameters(generator) + count_parameters(discriminator)
    print(f"Number of Generator Parameters: {num_generator_params/ 1000:.2f}K")
    print(f"Number of Discriminator Parameters: {num_discriminator_params/ 1000:.2f}K")
    print(f"Number of Parameters: {num_params/ 1000:.2f}K")
    # %%
    """training"""
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    train_module.train_function(
        generator=generator,
        discriminator=discriminator,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        train_data=train_dataset.data,
        data_sampler=data_sampler,
        transformer=train_dataset.EncodedInfo.transformer,
        config=config,
        mean=mean,
        std=std,
        device=device)
    # %%
    """model save"""
    base_name = f"{config['model']}_{config['latent_dim']}_{config['batch_size']}_{config['epochs']}_{config['generator_dim']}_{config['discriminator_dim']}_{config['dataset']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"
    torch.save(generator.state_dict(), f"./{model_dir}/{model_name}.pth")
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
