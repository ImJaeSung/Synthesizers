#%%
import os
#%%
import argparse
import importlib
#%%
import torch
from torch.utils.data import DataLoader
#%%
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import set_random_seed
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

project = "GOGGLE" # put your WANDB project name
# entity =  "wotjd1410" # put your WANDB username
run = wandb.init(
    project=project, 
    # entity=entity,
    tags=['inferenece'], # put tags of this python project
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
    parser.add_argument('--dataset', type=str, default='breast', 
                        help="""
                        Tabular dataset options: 
                        breast, banknote, default, whitewine, bankruptcy, BAF
                        """)
    parser.add_argument('--batch_size', default=64, type=int, 
                        help="Batch size for training")
    parser.add_argument('--lr', default=0.001, type=float, 
                        help="Learning rate")
    parser.add_argument('--alpha', default=0.1, type=float, 
                        help='Alpha value for GoggleLoss (KL divergence)')
    parser.add_argument('--beta', default=0.1, type=float, 
                        help='Beta value for GoggleLoss (Graph sparsity)')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    base_name = f"{config['dataset']}_{config['lr']}_{config['batch_size']}_{config['alpha']}_{config['beta']}"
    model_name = f"GOGGLE_{base_name}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model'
    )
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    

    config["cuda"] = torch.cuda.is_available()
    device = 'cpu'
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(
        config, 
        train=True
    )
    test_dataset = CustomDataset(
        config,
        cont_scalers=train_dataset.cont_scalers,
        train=False
    )
    #%%
    """model load"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.Goggle(config, device)
    model.to(device)
    model.train()
    #%%
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
    syndata = model.generate_synthetic_data(n, train_dataset)
    #%%
    results = evaluate(syndata, train_dataset, test_dataset, config, device)
    results = results._asdict()

    for x, y in results.items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()

    for x,y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    #%%
    wandb.config.update(config, allow_val_change=True)    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
