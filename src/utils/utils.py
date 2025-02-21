
import torch
import torch.nn.functional as F
from kron_torch import Kron

def parse_cnn_size(net_size):
    if net_size == "small":
        return (16, 32, 32)
    elif net_size == "medium":
        return (32, 64, 64)
    elif net_size == "large":
        return (64, 128, 128)
    
def parse_mlp_depth(net_size):
    if net_size == "small":
        return 1
    elif net_size == "medium":
        return 3
    elif net_size == "large":
        return 5
    elif net_size == "xlarge":
        return 10
    else:
        raise ValueError(f"Unknown network size: {net_size}")
    
def parse_mlp_width(net_size):
    if net_size == "small":
        return 512
    elif net_size == "medium":
        return 1024
    elif net_size == "large":
        return 2048
    elif net_size == "xlarge":
        return 4096
    
def get_act_fn_functional(act_fn):
    if act_fn == "relu":
        return F.relu
    elif act_fn == "tanh":
        return F.tanh
    elif act_fn == "elu":
        return F.elu
    elif act_fn == "selu":
        return F.selu
    elif act_fn == "leaky_relu":
        return F.leaky_relu
    elif act_fn == "sigmoid":
        return F.sigmoid
    elif act_fn == "silu":
        return F.silu
    else:
        raise ValueError(f"Unknown activation function: {act_fn}")

def get_act_fn_clss(act_fn):
    if act_fn == "relu":
        return torch.nn.ReLU
    elif act_fn == "tanh":
        return torch.nn.Tanh
    elif act_fn == "elu":
        return torch.nn.ELU
    elif act_fn == "selu":
        return torch.nn.SELU
    elif act_fn == "leaky_relu":
        return torch.nn.LeakyReLU
    elif act_fn == "sigmoid":
        return torch.nn.Sigmoid
    elif act_fn == "silu":
        return torch.nn.SiLU
    else:
        raise ValueError(f"Unknown activation function: {act_fn}")
    
def get_optimizer(optimizer):
    if optimizer == "adam":
        return torch.optim.Adam
    elif optimizer == "radam":
        return torch.optim.RAdam
    elif optimizer == "sgd":
        return torch.optim.SGD
    elif optimizer == "rmsprop":
        return torch.optim.RMSprop
    elif optimizer == "kron":
        return Kron
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}") 
