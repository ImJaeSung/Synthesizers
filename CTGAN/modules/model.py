# %%
"""
Reference:
[1] https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py
"""
# %%
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
)
import torch.nn.functional as F

# %%
class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        """CTGAN use the PacGAN framework with 10 samples in each pac to prevent mode collaspe."""
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(
        self,
        real_data,
        fake_data,
        device="cpu", 
        pac=10, 
        lambda_=10):

        """CTGAN is trained using WGAN loss with gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


# %%
class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


# %%
class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.
        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for _ in range(10):
                transformed = F.gumbel_softmax(
                    logits, tau=tau, hard=hard, eps=eps, dim=dim
                )
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
    
    def generate_synthetic_data(self, n, train_dataset, data_sampler, config, device):
        steps = n // config["batch_size"] + 1
        data = []
        for _ in tqdm(range(steps), desc="generate synthetic data..."):
            mean = torch.zeros(config["batch_size"], config["latent_dim"])
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(device)

            condvec = data_sampler.sample_original_condvec(config["batch_size"])
            c1 = condvec
            c1 = torch.from_numpy(c1).to(device)
            fakez = torch.cat([fakez, c1], dim=1)

            fake = self(fakez)
            fakeact = apply_activate(fake, train_dataset.EncodedInfo.transformer, self._gumbel_softmax)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        syndata = train_dataset.EncodedInfo.transformer.inverse_transform(data)
        
        # post-processing
        syndata[train_dataset.categorical_features] = syndata[train_dataset.categorical_features].astype(int)
        syndata[train_dataset.integer_features] = syndata[train_dataset.integer_features].round(0).astype(int)
        
        for feature in train_dataset.categorical_features:
            syndata[feature] = syndata[feature].apply(lambda x: 0 if x < 0 else x)

        return syndata

# %%
def apply_activate(data, transformer, gumbel_softmax):
    """Apply proper activation function to the output of the generator."""
    data_t = []
    st = 0
    for column_info in transformer.output_info_list:
        for span_info in column_info:
            if span_info.activation_fn == "tanh":
                ed = st + span_info.dim
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif span_info.activation_fn == "softmax":
                ed = st + span_info.dim
                transformed = gumbel_softmax(data[:, st:ed], tau=0.2)
                data_t.append(transformed)
                st = ed
            else:
                raise ValueError(
                    f"Unexpected activation function {span_info.activation_fn}."
                )

    return torch.cat(data_t, dim=1)


# %%
def validate_discrete_columns(train_data, discrete_columns):
    """Check whether ``discrete_columns`` exists in ``train_data``.
    Args:
        train_data (numpy.ndarray or pandas.DataFrame):
            Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
        discrete_columns (list-like):
            List of discrete columns to be used to generate the Conditional
            Vector. If ``train_data`` is a Numpy array, this list should
            contain the integer indices of the columns. Otherwise, if it is
            a ``pandas.DataFrame``, this list should contain the column names.
    """
    if isinstance(train_data, pd.DataFrame):
        invalid_columns = set(discrete_columns) - set(train_data.columns)
    elif isinstance(train_data, np.ndarray):
        invalid_columns = []
        for column in discrete_columns:
            if column < 0 or column >= train_data.shape[1]:
                invalid_columns.append(column)
    else:
        raise TypeError("``train_data`` should be either pd.DataFrame or np.array.")

    if invalid_columns:
        raise ValueError(f"Invalid columns found: {invalid_columns}")


# %%