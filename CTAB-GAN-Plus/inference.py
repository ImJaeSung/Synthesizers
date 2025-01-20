# %%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import importlib
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.simulation import set_random_seed
from evaluation.evaluation import evaluate
from evaluation import utility

import warnings
warnings.filterwarnings('ignore')
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

project = "distvae_journal_baseline" # put your WANDB project name
entity = "anseunghwan" # put your WANDB username

run = wandb.init(
    project=project, 
    entity=entity, 
    tags=["inference"], # put tags of this python project
)
# %%
import argparse

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
    parser = argparse.ArgumentParser("parameters")
    
    parser.add_argument("--ver", type=int, default=0, help="model version")
    parser.add_argument("--model", type=str, default="CTAB-GAN-Plus")
    parser.add_argument('--dataset', type=str, default='abalone', 
                        help="""
                        Dataset options: 
                        abalone, adult, banknote, breast, concrete, 
                        kings, loan, seismic, redwine, whitewine
                        """)
    
    parser.add_argument(
        "--latent_dim",
        default=100,
        type=int,
        help="size of the noise vector fed to the generator",
    )
    parser.add_argument('--SmallerGen', default=False, type=str2bool,
                        help='Use smaller batch size for synthetic data generation')

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

#%% 
def main():
    # %%
    config = vars(get_args(debug=False))  # default configuration

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
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    wandb.config.update(config)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    # %%
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset

    """dataset"""
    train_dataset = CustomDataset(
        config,
        train=True)
    test_dataset = CustomDataset(
        config,
        train=False)
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
    
    synthesizer = CTABGANSynthesizer(config, data_prep, train_dataset.EncodedInfo.type)
    # %%
    """model"""
    generator = synthesizer.generator
    
    if config["cuda"]:
        generator.load_state_dict(
            torch.load(
                model_dir + "/" + model_name
            )
        )
    else:
        generator.load_state_dict(
            torch.load(
                model_dir + "/" + model_name,
                map_location=torch.device("cpu"),
            )
        )
    generator.eval()
    #%%
    """synthetic dataset"""
    sample = synthesizer.sample(len(train_dataset.raw_data)) 
    syndata = data_prep.inverse_prep(sample)
    syndata[train_dataset.EncodedInfo.categorical_features] = syndata[train_dataset.EncodedInfo.categorical_features].astype(int)
    #%%
    """evaluation"""
    results = evaluate(syndata, train_dataset, test_dataset, config, device)
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    
    print("Marginal Distribution...")
    figs = utility.marginal_plot(train_dataset.raw_data, syndata, config, model_name)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%%
if __name__ == "__main__":
    main()
# %%