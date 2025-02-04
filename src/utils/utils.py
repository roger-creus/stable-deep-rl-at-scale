
import torch
import torch.nn.functional as F
from rl_act import RLAct, Meta_ADARL, Heuristic_ADARL, Smooth_Meta_ADARL

def parse_network_size(net_size):
    if net_size not in ["small", "medium", "large", "default"]: raise ValueError(f"Unknown network size: {net_size}")
    if net_size == "small":
        return (16, 32, 32), 512, 1
    elif net_size == "medium":
        return (32, 64, 64), 1024, 2
    elif net_size == "large":
        return (64, 128, 128), 2048, 3
    elif net_size == "default":
        return (32, 64, 64), 512, 1
    
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
    elif act_fn == "rl_act":
        return RLAct()
    elif act_fn == "meta_adarl":
        return Meta_ADARL()
    elif act_fn == "smooth_meta_adarl":
        return Smooth_Meta_ADARL()
    elif act_fn == "heuristic_adarl":
        return Heuristic_ADARL()
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
    elif act_fn == "rl_act":
        return RLAct
    elif act_fn == "meta_adarl":
        return Meta_ADARL
    elif act_fn == "smooth_meta_adarl":
        return Smooth_Meta_ADARL
    elif act_fn == "heuristic_adarl":
        return Heuristic_ADARL
    else:
        raise ValueError(f"Unknown activation function: {act_fn}")
    
def find_all_modules(module, module_clss=Meta_ADARL):
    """Recursively find all instances of Meta_ADARL in the module tree."""
    meta_modules = []
    for child in module.children():
        if isinstance(child, module_clss):
            meta_modules.append(child)
        else:
            meta_modules.extend(find_all_modules(child, module_clss))
    return meta_modules