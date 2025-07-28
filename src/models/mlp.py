import torch
import numpy as np
import torch.nn as nn
from utils.utils import get_act_fn_clss, get_act_fn_functional
from torch.nn.utils import spectral_norm

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def apply_weight_clipping(model, clip_value=1.0):
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)

class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([nn.functional.relu(x), nn.functional.relu(-x)], dim=-1)

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        last_act=False,
        use_ln=False,
        use_spectral_norm=False,
        activation_fn="relu",
        device='cpu',
        linear_clss=nn.Linear,
        weight_clip_value=None
    ):
        super(MLP, self).__init__()
        self.output_size = output_size
        self.weight_clip_value = weight_clip_value

        mlp = []
        if activation_fn.lower() == "crelu":
            act_ = CReLU
            activation_expand = 2
        else:
            act_ = get_act_fn_clss(activation_fn)
            activation_expand = 1

        if num_layers == 1:
            hidden_size = output_size

        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size * activation_expand
            out_dim = hidden_size if i < num_layers - 1 else output_size

            if linear_clss.__name__ == 'NoisyLinear':
                layer = linear_clss(in_dim, out_dim, device=device)
            else:
                layer = layer_init(linear_clss(in_dim, out_dim, device=device))

            if use_spectral_norm:
                layer = spectral_norm(layer)

            mlp.append(layer)

            if i < num_layers - 1:
                if use_ln:
                    mlp.append(nn.LayerNorm(out_dim, device=device))
                mlp.append(act_())
            elif i == num_layers - 1:
                if use_ln:
                    mlp.append(nn.LayerNorm(out_dim, device=device))
                if last_act:
                    mlp.append(act_())

        self.net = nn.Sequential(*mlp)

    def forward(self, x):
        out = self.net(x)
        if self.weight_clip_value is not None:
            apply_weight_clipping(self, self.weight_clip_value)
        return out
