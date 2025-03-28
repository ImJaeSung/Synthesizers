# %%
from tqdm import tqdm
import wandb
import numpy as np

import torch
import torch.nn.functional as F
# %%
def train_function(
    output_info_list,
    dataloader, 
    model, 
    config, 
    optimizer, 
    device):
    for epoch in range(config["epochs"]):
        logs = {
            "loss": [],
            "recon": [],
            "KL": [],
        }
        # for debugging
        logs["activated"] = []

        for x_batch in tqdm(dataloader, desc="inner loop..."):
            x_batch = x_batch.to(device)

            # with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()

            mean, logvar, _, xhat = model(x_batch)

            loss_ = []

            """reconstruction"""
            start = 0
            recon = 0
            for column_info in output_info_list:
                for span_info in column_info:
                    if span_info.activation_fn != "softmax":
                        end = start + span_info.dim
                        std = model.sigma[start]
                        residual = x_batch[:, start] - torch.tanh(xhat[:, start])
                        recon += (residual**2 / 2 / (std**2)).mean()
                        recon += torch.log(std)
                        start = end
                    else:
                        end = start + span_info.dim
                        recon += F.cross_entropy(
                            xhat[:, start:end],
                            torch.argmax(x_batch[:, start:end], dim=-1),
                            reduction="mean",
                        )
                        start = end
            loss_.append(("recon", recon))

            """KL-Divergence"""
            KL = torch.pow(mean, 2).sum(dim=1)
            KL -= logvar.sum(dim=1)
            KL += torch.exp(logvar).sum(dim=1)
            KL -= config["latent_dim"]
            KL *= 0.5
            KL = KL.mean()
            loss_.append(("KL", KL))

            ### activated: for debugging
            var_ = torch.exp(logvar) < 0.1
            loss_.append(('activated', var_.float().mean()))

            loss = config["loss_factor"] * recon + KL
            loss_.append(("loss", loss))

            loss.backward()
            optimizer.step()
            # model.sigma.data.clamp_(0.01, 0.1)
            model.sigma.data.clamp_(config["sigma_range"][0], config["sigma_range"][1])

            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
                
        if epoch % 10 == 0:                
            print_input = "[epoch {:03d}]".format(epoch + 1)
            print_input += "".join(
                [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
            )
            print(print_input)

        """update log"""
        wandb.log({x: np.mean(y) for x, y in logs.items()})

    return
# %%
