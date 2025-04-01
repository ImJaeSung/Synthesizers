"""Reference:
[1] https://github.com/ChaejeongLee/CoDi/blob/main/models/tabular_unet.py
[2] https://github.com/ChaejeongLee/CoDi/blob/main/models/layers.py
"""
#%%
import math
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
#%%
class Encoder(nn.Module):
  def __init__(self, config, Continuous):
    super(Encoder, self).__init__()
    encoder_dim = config["Cont_encoder_dim"] if Continuous else config["Disc_encoder_dim"]
    embed_dim = config["embed_dim_Cont"] if Continuous else config["embed_dim_Disc"]
    
    self.encoding_blocks = nn.ModuleList()
    for i in range(len(encoder_dim)):
      if (i+1) == len(encoder_dim): break
      encoding_block = EncodingBlock(
          encoder_dim[i], encoder_dim[i+1], embed_dim * 4, config)
      self.encoding_blocks.append(encoding_block)

  def forward(self, x, t):
    skip_connections = []
    for encoding_block in self.encoding_blocks:
      x, skip_connection = encoding_block(x, t)
      skip_connections.append(skip_connection)
    return skip_connections, x

class EncodingBlock(nn.Module):
  def __init__(self, input_dim, output_dim, embed_dim, config):
    super(EncodingBlock, self).__init__()
    self.layer1 = nn.Sequential( 
        nn.Linear(input_dim, output_dim),
        get_activation_fn(config["activation_fn"])
    ) 
    self.temb_proj = nn.Sequential(
        nn.Linear(embed_dim, output_dim),
        get_activation_fn(config["activation_fn"])
    )
    self.layer2 = nn.Sequential(
        nn.Linear(output_dim, output_dim),
        get_activation_fn(config["activation_fn"])
    )
    
  def forward(self, x, t):
    x = self.layer1(x).clone()
    x += self.temb_proj(t)
    x = self.layer2(x)
    skip_connection = x
    return x, skip_connection

#%%
class Decoder(nn.Module):
  def __init__(self, config, Continuous):
    super(Decoder, self).__init__()
    encoder_dim = config["Cont_encoder_dim"] if Continuous else config["Disc_encoder_dim"]
    decoder_dim = list(reversed(encoder_dim))
    
    embed_dim = config["embed_dim_Cont"] if Continuous else config["embed_dim_Disc"]
    
    self.decoding_blocks = nn.ModuleList()
    for i in range(len(decoder_dim)):
      if (i+1)==len(decoder_dim): break
      decoding_block = DecodingBlock(
          decoder_dim[i], decoder_dim[i+1], embed_dim * 4, config)
      self.decoding_blocks.append(decoding_block)

  def forward(self, skip_connections, x, t):
    zipped = zip(reversed(skip_connections), self.decoding_blocks)
    for skip_connection, decoding_block in zipped:
      x = decoding_block(skip_connection, x, t)
    return x

class DecodingBlock(nn.Module):
  def __init__(self, input_dim, output_dim, embed_dim, config):
    super(DecodingBlock, self).__init__()
    self.layer1 = nn.Sequential( 
        nn.Linear(input_dim * 2, input_dim),
        get_activation_fn(config["activation_fn"])
    )
    self.temb_proj = nn.Sequential(
        nn.Linear(embed_dim, input_dim),
        get_activation_fn(config["activation_fn"])
    )
    self.layer2 = nn.Sequential(
        nn.Linear(input_dim, output_dim),
        get_activation_fn(config["activation_fn"])
    )
    
  def forward(self, skip_connection, x, t):
    
    x = torch.cat((skip_connection, x), dim=1)
    x = self.layer1(x).clone()
    x += self.temb_proj(t)
    x = self.layer2(x)

    return x
#%%
class tabularUnet(nn.Module):
  def __init__(self, config, train_dataset_Cont, train_dataset_Disc, Continuous=True):
    super().__init__()

    self.encoder_dim = config["Cont_encoder_dim"] if Continuous else config["Disc_encoder_dim"]
    self.embed_dim = config["embed_dim_Cont"] if Continuous else config["embed_dim_Disc"]
    
    self.activation_fn = get_activation_fn(config["activation_fn"])
    input_size = train_dataset_Cont.shape[1] if Continuous else train_dataset_Disc.shape[1]
    condition_size = train_dataset_Disc.shape[1] if Continuous else train_dataset_Cont.shape[1]
    
    modules = []
    
    modules.append(nn.Linear(self.embed_dim, self.embed_dim * 4))
    modules[-1].weight.data = default_init()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    
    modules.append(nn.Linear(self.embed_dim*4, self.embed_dim * 4))
    modules[-1].weight.data = default_init()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)

    # cond = condition_size
    condition_output_size = (input_size)//2
    if condition_output_size < 2:
      condition_output_size = input_size
    modules.append(nn.Linear(condition_size, condition_output_size))
    modules[-1].weight.data = default_init()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)

    self.all_modules = nn.ModuleList(modules)

    """Encoder"""
    # input_dim = train_dataset_Cont.shape[1] + train_dataset_Disc.shape[1]
    input_dim = input_size + condition_output_size 
    self.inputs = nn.Linear(
        input_dim, self.encoder_dim[0]
    ) # input + condition -> encoder

    self.encoder = Encoder(config, Continuous=Continuous) # encoder

    """Bottom"""
    self.bottom_block = nn.Linear(
        self.encoder_dim[-1], self.encoder_dim[-1]) 
    
    """Decoder"""
    self.decoder = Decoder(config, Continuous=Continuous)

    output_dim = train_dataset_Cont.shape[1] if Continuous else train_dataset_Disc.shape[1]
    self.outputs = nn.Linear(self.encoder_dim[0], output_dim)

  def forward(self, x, time_condition, condition):
    x = x.float()
    condition = condition.float()
    
    modules = self.all_modules 
    m_idx = 0

    # time embedding
    time_embed = get_timestep_embedding(time_condition, self.embed_dim)
    time_embed = modules[m_idx](time_embed)
    m_idx += 1
    time_embed = self.activation_fn(time_embed)
    time_embed = modules[m_idx](time_embed)
    m_idx += 1
    
    # condition layer
    condition = modules[m_idx](condition)
    m_idx += 1
    x = torch.cat([x, condition], dim=1).float()
    inputs = self.inputs(x) # input layer
    skip_connections, encoding = self.encoder(inputs, time_embed)
    encoding = self.bottom_block(encoding)
    encoding = self.activation_fn(encoding)
   
    x = self.decoder(skip_connections, encoding, time_embed) 
    outputs = self.outputs(x)

    return outputs
#%%
def generate_synthetic_data(train_dataset, Sampler_Cont, Trainer_Disc, config, device):
    # data = []
    # steps = n // self.config["batch_size"] + 1
    C = train_dataset.num_continuous_features

    train_dataset_Cont = train_dataset.data[:, :C]
    train_dataset_Disc = train_dataset.data[:, C:]
    
    with torch.no_grad():
        x_T_Cont = torch.randn(train_dataset_Cont.shape[0], train_dataset_Cont.shape[1]).to(device)
        log_x_T_Disc = log_sample_categorical(torch.zeros(train_dataset_Disc.shape, device=device), train_dataset.num_categories).to(device)
        
        x_Cont, x_Disc = sampling_with(x_T_Cont, log_x_T_Disc, Sampler_Cont, Trainer_Disc, config)
        x_Disc = apply_activate(x_Disc, train_dataset.EncodedInfo_list[C:])
    
    """categorical"""
    st = 0
    tmp_x_Disc = []
    for column_info in train_dataset.EncodedInfo_list[C:]:
        ed = st + column_info[0]  
        tmp_x_Disc.append(torch.argmax(x_Disc[:, st:ed], dim=-1).unsqueeze(1))
        
        st = ed   
    
    x_Disc = torch.cat(tmp_x_Disc, dim=1)    
    
    data = torch.cat([x_Cont, x_Disc], dim=1)
    data = pd.DataFrame(data.cpu().numpy(), columns=train_dataset.continuous_features + train_dataset.categorical_features)
 
    """continuous"""
    data[train_dataset.continuous_features] = (data[train_dataset.continuous_features]+ 1) / 2
    data[train_dataset.continuous_features] = data[train_dataset.continuous_features]  * (train_dataset.max - train_dataset.min) + train_dataset.min
    
    """post-process integer columns (calibration)"""
    data[train_dataset.integer_features] = data[train_dataset.integer_features].round(0).astype(int)
    data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)

    return data
#%%
def log_sample_categorical(logits, num_classes):
    full_sample = []
    k = 0
    for i in range(len(num_classes)):
        logits_column = logits[:, k:num_classes[i]+k]
        k += num_classes[i]
        uniform = torch.rand_like(logits_column)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits_column).argmax(dim=1)
        col_t = np.zeros(logits_column.shape)
        col_t[np.arange(logits_column.shape[0]), sample.detach().cpu()] = 1
        full_sample.append(col_t)
        
    full_sample = torch.tensor(np.concatenate(full_sample, axis=1))
    log_sample = torch.log(full_sample.float().clamp(min=1e-30))
    return log_sample

#%%
def sampling_with(x_T_Cont, log_x_T_Disc, sampler_Cont, Trainer_Disc, config):
    x_t_Cont = x_T_Cont
    x_t_Disc = log_x_T_Disc

    for timestep_ in reversed(range(config["diffusion_steps"])):
        timestep = x_t_Cont .new_ones([x_t_Cont .shape[0], ], dtype=torch.long) * timestep_
        mean, log_var = sampler_Cont.p_mean_variance(
            x_t=x_t_Cont, timestep=timestep, condition=x_t_Disc.to(x_t_Cont.device))
        
        if timestep_ > 0:
            noise = torch.randn_like(x_t_Cont)
        elif timestep_ == 0:
            noise = 0
            
        x_t_minus_1_Cont = mean + torch.exp(0.5 * log_var) * noise
        x_t_minus_1_Cont = torch.clip(x_t_minus_1_Cont, -1., 1.)
        
        x_t_minus_1_Disc = Trainer_Disc.p_sample(x_t_Disc, timestep, x_t_Cont)

    return x_t_minus_1_Cont, x_t_minus_1_Disc
#%%
def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'sigmoid':
            ed = st + item[0]
            data_t.append(data[:,st:ed])
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.softmax(data[:, st:ed]))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)
#%%
def get_activation_fn(activation_fn):
  if activation_fn == 'elu':
    return nn.ELU()
  elif activation_fn == 'relu':
    return nn.ReLU()
  elif activation_fn == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif activation_fn == 'swish':
    return nn.SiLU()
  elif activation_fn == 'tanh':
    return nn.Tanh()
  elif activation_fn == 'softplus':
    return nn.Softplus()
  else:
    raise NotImplementedError('activation function does not exist!')

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  
  half_dim = embedding_dim // 2
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1: 
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

# %%
