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
        
        
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, activation_fn, use_ln, device):
        super(ResidualBlock, self).__init__()
        self.linear = layer_init(nn.Linear(hidden_size, hidden_size, device=device))
        self.use_ln = use_ln
        if self.use_ln:
            self.ln = nn.LayerNorm(hidden_size, device=device)
        self.activation = activation_fn()
        
    def forward(self, x):
        residual = x
        out = self.linear(x)
        if self.use_ln:
            out = self.ln(out)
        out = self.activation(out)
        return residual + out

class ResidualMLP(nn.Module):
    """
    A multi-layer perceptron with residual connections.
    
    All modules are composed into a single nn.Sequential object called self.net.
    
    For num_layers == 1:
      A single linear layer (optionally with layer norm and activation) mapping input -> output.
    
    For num_layers >= 2:
      - Input layer: maps input -> hidden representation.
      - Residual Blocks: (num_layers - 2) blocks operating in hidden_size.
      - Output layer: maps hidden representation -> output.
      - Optionally applies layer normalization and activation at the output.
    """
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
        super(ResidualMLP, self).__init__()
        act_fn = get_act_fn_clss(activation_fn)
        layers = []
        
        if num_layers == 1:
            # Single layer: directly map input to output.
            layers.append(
                layer_init(nn.Linear(input_size, output_size, device=device))
            )
            if use_ln:
                layers.append(nn.LayerNorm(output_size, device=device))
            if last_act:
                layers.append(act_fn())
        else:
            # Input layer: map input to hidden representation.
            layers.append(
                layer_init(nn.Linear(input_size, hidden_size, device=device))
            )
            
            # TODO: should we do this? i.e. add layer norm before residual blocks.
            if use_ln:
                layers.append(nn.LayerNorm(hidden_size, device=device))
            
            # Add residual blocks. (num_layers - 2) blocks.
            for _ in range(num_layers - 2):
                layers.append(
                    ResidualBlock(hidden_size, act_fn, use_ln, device)
                )
                
            # Output layer: map hidden representation to output.
            layers.append(
                layer_init(nn.Linear(hidden_size, output_size, device=device))
            )
            if use_ln:
                layers.append(nn.LayerNorm(output_size, device=device))
            if last_act:
                layers.append(act_fn())
        
        # All layers are combined into a single nn.Sequential object.
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
