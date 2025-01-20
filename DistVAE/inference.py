#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import json
import argparse
import importlib
#%%
import torch
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import interpolate

#%%
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.simulation import set_random_seed
from evaluation.evaluation import evaluate
#%%
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

run = wandb.init(
    project="DistVAE.ver3", # put your WANDB project name
    # entity="", # put your WANDB username
    tags=['Inference.ver2'], # put tags of this python project
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
                        help='Dataset options: abalone, adult, banknote, breast, cabs, concrete, covtype, credit, kings, letter, loan, redwine, spam, whitewine, yeast')
    parser.add_argument('--beta', default=0.5, type=float,
                        help='observation noise')
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
    model_name = f"DistVAE_beta{config['beta']}_{config['dataset']}"
    if config["SmallerGen"]:
        model_name = "small_" + model_name

    artifact = wandb.use_artifact(
        f"DistVAE.ver3/{model_name}:v{config['ver']}",
        type='model'
    )
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()

    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    
    # if not os.path.exists('./assets/{}'.format(config["dataset"])):
    #     os.makedirs('./assets/{}'.format(config["dataset"]))
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.{}_dataset'.format(config["dataset"]))
    CustomDataset = dataset_module.CustomDataset
    
    train_dataset = CustomDataset(
        config,
        train=True
    )
    test_dataset = CustomDataset(
        config,
        train=False
    )
    
    EncodedInfo_list = train_dataset.EncodedInfo_list
    CRPS_dim = sum([x.dim for x in EncodedInfo_list if x.activation_fn == 'CRPS'])
    softmax_dim = sum([x.dim for x in EncodedInfo_list if x.activation_fn == 'softmax'])
    config["CRPS_dim"] = CRPS_dim
    config["softmax_dim"] = softmax_dim
    #%%
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)

    model = model_module.VAE(config, device).to(device)

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
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params/1000:.2f}k")
    wandb.log({"Number of Parameters": num_params/1000})
    #%%
    """Synthetic Data Generation"""
    n = len(train_dataset.raw_data)
    syndata = model.generate_synthetic_data(n, EncodedInfo_list, train_dataset)
    #%%
    results, fig = evaluate(syndata, train_dataset, test_dataset, config)

    for x,y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    # wandb.log({'Marginal Histogram': wandb.Image(fig)})
    #%%
    wandb.config.update(config, allow_val_change=True)    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()

# %%
