import torch
import numpy as np
import torch.nn as nn
from utils.utils import get_act_fn_clss

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        last_act=True,
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super(MLP, self).__init__()
        mlp = []
        act_ = get_act_fn_clss(activation_fn)
        
        if num_layers == 1: hidden_size = output_size
        
        for i in range(num_layers):
            # Input layer
            if i == 0:
                mlp.append(
                    layer_init(nn.Linear(input_size, hidden_size, device=device))
                )
            # Output layer
            elif i == num_layers - 1:
                mlp.append(
                    layer_init(nn.Linear(hidden_size, output_size, device=device))
                )
            # Hidden layers
            else:
                mlp.append(
                    layer_init(nn.Linear(hidden_size, hidden_size, device=device))
                )
                
            if i < num_layers - 1:
                # Add a layer normalization layer if needed
                if use_ln:
                    mlp.append(
                        nn.LayerNorm(hidden_size, device=device)
                    )
                
                # Add a ReLU activation
                mlp.append(
                    act_()
                )
            elif i == num_layers - 1:
                if use_ln:
                    mlp.append(
                        nn.LayerNorm(output_size, device=device)
                    )
                
                if last_act:
                    mlp.append(act_())
            
        self.net = nn.Sequential(*mlp)
        
    def forward(self, x):
        return self.net(x)
        