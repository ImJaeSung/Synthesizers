#%%
import os
import time
import json 

import numpy as np
import argparse
import importlib

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from modules.model1 import Encoder_model, Decoder_model

from models.utils import set_random_seed, get_data_scaler, get_data_inverse_scaler
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "2stage_basline" # put your WANDB project name
# entity = "" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    ### default configs
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument("--model", type=str, default="STaSy")
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        Tabular dataset options: 
                        banknote, whitewine, breast, bankruptcy, musk,
                        abalone, anuran, shoppers, default, magic
                        """)
    parser.add_argument('--test_size', type=float, default=0.2)
    
    ### data
    parser.add_argument('--data.image_size',type = int, default=77)
    parser.add_argument('--data.centered', type=bool, default=False, help='Data centered')

    ### model
    parser.add_argument('--model.name', type=str, default="ncsnpp_tabular")
    parser.add_argument('--model.layer_type', type=str, default="concatsquash")
    parser.add_argument('--model.scale_by_sigma', type=bool, default=False)
    parser.add_argument('--model.ema_rate', type=float, default=0.9999)
    parser.add_argument('--model.activation', type=str, default="elu")
    parser.add_argument('--model.nf', type=int, default=64)
    parser.add_argument('--model.hidden_dims', type=list, default=[1024, 2048, 1024, 1024])
    parser.add_argument('--model.conditional', type=bool, default=True)
    parser.add_argument('--model.embedding_type', type=str, default="fourier")
    parser.add_argument('--model.fourier_scale', type=int, default=16)
    parser.add_argument('--model.conv_size', type=int, default=3)

    parser.add_argument('--model.sigma_min', type=float, default=0.01, help='Minimum sigma')
    parser.add_argument('--model.sigma_max', type=float, default=10., help='Maximum sigma')
    parser.add_argument('--model.num_scales', type=int, default=50)
    parser.add_argument('--model.alpha0', type=float, default=0.3)
    parser.add_argument('--model.beta0', type=float, default=0.95)
    ### test
    parser.add_argument('--test.n_iter', type=int, default=1)

    ### optim
    parser.add_argument('--optim.lr', type=float, default=2e-3)
    parser.add_argument('--optim.eps', type=float, default=1e-8, help='Epsilon value')
    parser.add_argument('--optim.beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--optim.weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--optim.optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--optim.warmup', type=int, default=5000)
    parser.add_argument('--optim.grad_clip', type=float, default=1.0)
    
    ### training
    ## default
    parser.add_argument('--training.epoch', type=int, default=10000) #10000 for paper setting
    parser.add_argument('--training.snapshot_freq', type=int, default=300)
    parser.add_argument('--training.eval_freq', type=int, default=100)
    parser.add_argument('--training.snapshot_freq_for_preemption', type=int, default=100)
    parser.add_argument('--training.snapshot_sampling', type=bool, default=True)
    parser.add_argument('--training.likelihood_weighting', type=bool, default=False)
    parser.add_argument('--training.continuous', type=bool, default=True)
    parser.add_argument('--training.eps', type=float, default=1e-05)
    parser.add_argument('--training.loss_weighting', type=bool, default=False)
    parser.add_argument('--training.spl', type=bool, default=True)
    parser.add_argument('--training.lambda_', type=float, default=0.5)

    ## specific
    parser.add_argument('--training.sde', type=str, default="vesde")
    parser.add_argument('--training.reduce_mean', type=bool, default=True)
    parser.add_argument('--training.n_iters', type=int, default=100000)
    parser.add_argument('--training.tolerance', type=float, default=1e-03)
    parser.add_argument('--training.hutchinson_type', type=str, default="Rademacher")
    parser.add_argument('--training.retrain_type', type=str, default="median")
    parser.add_argument('--training.batch_size', type=int, default=1000)

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    config['device'] = device
    wandb.config.update(config)
    #%%
    """dataset"""
    dataset_module = importlib.import_module(f"dataset.preprocess")
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    
    train_dataset = CustomDataset(
        config, 
        train=True)
    test_dataset = CustomDataset(
        config, 
        train=False,
        cont_scalers=train_dataset.cont_scalers, 
        cat_scalers=train_dataset.cat_scalers)     
    
    train_z = np.concatenate([train_dataset.X_num, train_dataset.X_cat], axis=1)
    config["data.image_size"] = train_z.shape[1]
   
    train_dataloader = DataLoader(
        train_z, 
        batch_size=config['training.batch_size'], 
        shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['training.batch_size'],
        shuffle=False)

    """model"""
    from models import utils as mutils
    score_model = mutils.create_model(config)

    from models.ema import ExponentialMovingAverage
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config["model.ema_rate"])

    # optimizer
    import losses as losses
    optimizer = losses.get_optimizer(config, score_model.parameters())
    
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

    initial_step = int(state['epoch'])

    batch_size = config["training.batch_size"] 

    shuffle_buffer_size = 10000
    num_epochs = None 

    scaler = get_data_scaler(config) 
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    import sde_lib as sde_lib
    if config["training.sde"].lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config["model.beta_min"], beta_max=config["model.beta_max"], N=config["model.num_scales"])
        sampling_eps = 1e-3
    elif config["training.sde"].lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config["model.beta_min"], beta_max=config["model.beta_max"], N=config["model.num_scales"])
        sampling_eps = 1e-3
    elif config["training.sde"].lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config["model.sigma_min"], sigma_max=config["model.sigma_max"], N=config["model.num_scales"])
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config['training.sde']} unknown.")
        logging.info(score_model)

    optimize_fn = losses.optimization_manager(config)
    continuous = config["training.continuous"]
    reduce_mean = config["training.reduce_mean"]
    likelihood_weighting = config["training.likelihood_weighting"]

    
    """number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params = count_parameters(score_model)
    print("the number of parameters", num_params)
    wandb.log({"the number of parameters": num_params})

    """checkpoint"""
    base_name = f"{config['model']}_{config['dataset']}_{config['model.sigma_min']}_{config['model.sigma_max']}_{config['optim.lr']}_{config['model.beta0']}_{config['model.alpha0']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"
    
    """training"""
    train_step_fn = losses.get_step_fn(
        sde, 
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean, 
        continuous=continuous,
        likelihood_weighting=likelihood_weighting, 
        workdir=model_dir, spl=config["training.spl"], 
        alpha0=config["model.alpha0"], 
        beta0=config["model.beta0"]
    )
    
    best_loss = np.inf
    for epoch in range(initial_step, config["training.epoch"]+1):
        start_time = time.time()
        state['epoch'] += 1
        batch_loss = 0
        batch_num = 0
        for iteration, batch in enumerate(train_dataloader): 
            batch = batch.to(config["device"]).float()

            num_sample = batch.shape[0]
            batch_num += num_sample
            loss = train_step_fn(state, batch)

            batch_loss += loss.item() * num_sample

        batch_loss = batch_loss / batch_num
        wandb.log({'loss' : batch_loss})
        if epoch % 200 == 0:
            print("epoch: %d, iter: %d, training_loss: %.5e" % (epoch, iteration, batch_loss))

        from utils import save_checkpoint, restore_checkpoint, apply_activate
        if batch_loss < best_loss:
            best_loss = batch_loss
            best_state = save_checkpoint(
                f"./{model_dir}/{model_name}.pth", state)

        end_time = time.time()

    """model save"""
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]),  
        type='model',
        metadata=config
    )
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./dataset/preprocess.py')
    wandb.log_artifact(artifact)    
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%
    
    