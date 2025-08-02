
import torch
import torch.nn.functional as F
import torch.nn as nn
from kron_torch import Kron
import torch_optimizer as optim

def parse_cnn_size(net_size):
    if net_size == "small":
        return (16, 32, 32)
    elif net_size == "medium":
        return (32, 64, 64)
    elif net_size == "large":
        return (64, 128, 128)
    
def parse_mlp_depth(net_size, net_type):
    if net_size == "small":
        return 1 if "residual" in net_type else (1*2) + 2
    elif net_size == "medium":
        return 3 if "residual" in net_type else (3*2) + 2
    elif net_size == "large":
        return 5 if "residual" in net_type else (5*2) + 2
    elif net_size == "xlarge":
        return 10 if "residual" in net_type else (10*2) + 2
    else:
        raise ValueError(f"Unknown network size: {net_size}")
    
def parse_mlp_width(net_size):
    if net_size == "small":
        return 512
    elif net_size == "medium":
        return 512 * 3
    elif net_size == "large":
        return 512 * 5
    elif net_size == "xlarge":
        return 512 * 10
    
def get_mlp_num_params(mlp_width, mlp_depth):
    neurons_per_layer = parse_mlp_width(mlp_width)
    num_layers = parse_mlp_depth(mlp_depth, "default")
    params = 3136 * neurons_per_layer + neurons_per_layer
    params += (num_layers - 1) * (neurons_per_layer * neurons_per_layer + neurons_per_layer)
    params += neurons_per_layer
    return params

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
    elif act_fn == "gelu":
        return F.gelu
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
    elif act_fn == "gelu":
        return torch.nn.GELU
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
    elif optimizer == "shampoo":
        return optim.Shampoo
    elif optimizer == "apollo":
        return optim.Apollo
    elif optimizer == "adabelief":
        return optim.AdaBelief
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}") 

def clean_grad_stats(grad_stats, use_ln=True):
    cleaned = {k.replace("network.", "").replace(".weight", "").replace(".", "_"): v for k, v in grad_stats.items()}
    
    if "ln" in cleaned.keys():
        cleaned = {k: v for k, v in cleaned.items() if "ln" not in k}
    elif use_ln:
        for k in list(cleaned.keys()):
            if "cnn" in k:
                try:
                    if int(k.split("_")[1]) % 3 != 0:
                        cleaned.pop(k)
                except:
                    if int(k.split("_")[2]) % 3 != 0:
                        cleaned.pop(k)
    
    cleaned_final = {f"{k.split('_')[0]}_{i}": v for i, (k, v) in enumerate(cleaned.items())}
    new_clean = {}
    c = 0
    for k, v in cleaned_final.items():
        if "trunk" in k:
            new_clean[f"mlp_{c}"] = v
            c += 1
        elif "q" in k:
            new_clean["q"] = v
        else:
            new_clean[k] = v
    return new_clean

def get_weight_norms(agent, use_ln=True):
    layer_weight_norms = {}
    for name, param in agent.named_parameters():
        if "weight" in name:
            if len(param.shape) == 1:
                continue
            weight_flat = param.detach().view(-1)
            weight_norm = weight_flat.norm().item()
            layer_weight_norms[name] = weight_norm
    return clean_grad_stats(layer_weight_norms, use_ln)

def get_dormant_neurons(agent, images_batch, use_ln=True):
    dormant_neurons = {}
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    hooks = []
    for name, module in agent.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        agent(images_batch)
    
    for name, activation in activations.items():
        if len(activation.shape) == 4:
            reshaped = activation.permute(1, 0, 2, 3).reshape(activation.shape[1], -1)
            dormant_count = (reshaped.max(dim=1)[0] <= 0).sum().item()
            dormant_neurons[name] = dormant_count / activation.shape[1]
        elif len(activation.shape) == 2:
            reshaped = activation.t()
            dormant_count = (reshaped.max(dim=1)[0] <= 0).sum().item()
            dormant_neurons[name] = dormant_count / activation.shape[1]
    
    for hook in hooks:
        hook.remove()
    
    return clean_grad_stats(dormant_neurons, use_ln)


def get_grad_norms(agent, use_ln=True):
    layer_grad_norms = {}
    for name, param in agent.named_parameters():
        if param.grad is not None and "weight" in name:
            if len(param.shape) == 1:
                continue
            grad_flat = param.grad.detach().view(-1)
            grad_norm = grad_flat.norm().item()
            layer_grad_norms[name] = grad_norm
    return clean_grad_stats(layer_grad_norms, use_ln)

def get_grad_cosine(agent, use_ln=True):
    grad_direction_cosine = {}
    for name, param in agent.named_parameters():
        if param.grad is not None and "weight" in name:
            if len(param.shape) == 1:
                continue
            
            grad_flat = param.grad.detach().view(-1)
            grad_norm = grad_flat.norm().item()
            grad_unit = grad_flat / grad_norm if grad_norm > 0 else grad_flat
            
            if name in agent.prev_grad_dirs:
                prev_grad_unit = agent.prev_grad_dirs[name]
                cosine_sim = torch.dot(grad_unit, prev_grad_unit).item()
                grad_direction_cosine[name] = cosine_sim
            else:
                grad_direction_cosine[name] = None
            
            agent.prev_grad_dirs[name] = grad_unit.clone()
            
    return clean_grad_stats(grad_direction_cosine, use_ln)