# %%
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
import importlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

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

project = "distvae_journal_baseline1" # put your WANDB project name
entity = "anseunghwan" # put your WANDB username

run = wandb.init(
    project=project, 
    entity=entity, 
    # tags=[""], # put tags of this python project
)
# %%
import argparse
import ast

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v


def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument(
        "--seed", type=int, default=1, help="seed for repeatable results"
    )
    parser.add_argument("--model", type=str, default="CTAB-GAN-Plus")
    parser.add_argument('--dataset', type=str, default='banknote', 
                        help="""
                        Tabular dataset options: 
                        banknote, whitewine, breast, bankruptcy, musk, madelon
                        """)
    
    parser.add_argument(
        "--latent_dim",
        default=100,
        type=int,
        help="size of the noise vector fed to the generator",
    )
    
    parser.add_argument(
        "--num_channels",
        default=64,
        type=int,
        help="no. of channels for deciding respective hidden layers of discriminator and generator networks",
    )
    parser.add_argument(
        "--class_dim",
        default=[256, 256, 256, 256],
        type=arg_as_list,
        help="list containing dimensionality of hidden layers for the classifier network",
    )
    
    # optimization options
    parser.add_argument(
        "--test_size", default=0.2, type=float, help="split train and test"
    )
    parser.add_argument(
        "--epochs", default=150, type=int, help="no. of epochs to train the model"
    )
    parser.add_argument(
        "--batch_size",
        default=500,
        type=int,
        help="no. of records to be processed in each mini-batch of training",
    )
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument(
        "--weight_decay",
        default=1e-5,
        type=float,
        help="parameter to decide strength of regularization of the network based on constraining l2 norm of weights",
    )
    parser.add_argument('--SmallerGen', default=False, type=str2bool,
                        help='Use smaller batch size for synthetic data generation')


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

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    # %%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(
        config,
        train=True)
    # %%
    dataprep_module = importlib.import_module('model.pipeline.data_preparation')
    importlib.reload(dataprep_module)
    DataPrep = dataprep_module.DataPrep
    
    synthesizer_module = importlib.import_module('model.synthesizer.ctabgan_synthesizer')
    importlib.reload(synthesizer_module)
    CTABGANSynthesizer = synthesizer_module.CTABGANSynthesizer
    
    data_prep = DataPrep(
        raw_df=train_dataset.data,
        EncodedInfo=train_dataset.EncodedInfo,
        test_ratio=config['test_size'])
    
    synthesizer = CTABGANSynthesizer(
        config, data_prep, train_dataset.EncodedInfo.type, device)
    
    synthesizer.fit()
    # %%
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params = count_parameters(synthesizer.generator) + count_parameters(synthesizer.discriminator) + count_parameters(synthesizer.classifier)
    print(f"Number of Parameters: {num_params}")
    # %%
    """model save"""
    base_name = f"{config['model']}_{config['latent_dim']}_{config['dataset']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"
    torch.save(synthesizer.generator.state_dict(), f"./{model_dir}/{model_name}.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config) 
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./datasets/preprocess.py')
    artifact.add_file("./model/synthesizer/ctabgan_synthesizer.py")
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
# %%
if __name__ == "__main__":
    main()
# %%
