import torch
import numpy as np
import torch.nn as nn
from utils.utils import get_act_fn_clss, get_act_fn_functional

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
        last_act=False,
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super(MLP, self).__init__()
        self.output_size = output_size
        
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
        
class ResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super().__init__()
        self.use_ln = use_ln
        self.act_ = get_act_fn_functional(activation_fn)
        
        self.linear0 = layer_init(nn.Linear(hidden_size, hidden_size, device=device))
        if self.use_ln:
            self.ln0 = nn.LayerNorm(hidden_size, device=device)
            
        self.linear1 = layer_init(nn.Linear(hidden_size, hidden_size, device=device))
        if self.use_ln:
            self.ln1 = nn.LayerNorm(hidden_size, device=device)

    def forward(self, x):
        residual = x
        x = self.act_(x)
        x = self.linear0(x)
        if self.use_ln:
            x = self.ln0(x)
        x = self.act_(x)
        x = self.linear1(x)
        if self.use_ln:
            x = self.ln1(x)
        return x + residual

class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        last_act=False,
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super().__init__()
        self.output_size = output_size
        act_fn_class = get_act_fn_clss(activation_fn)
        layers = []
        
        layers.append(layer_init(nn.Linear(input_size, hidden_size, device=device)))
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size, device=device))
        
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_size, use_ln=use_ln, activation_fn=activation_fn, device=device))
        
        layers.append(layer_init(nn.Linear(hidden_size, output_size, device=device)))
        if use_ln:
            layers.append(nn.LayerNorm(output_size, device=device))
        if last_act:
            layers.append(act_fn_class())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
    
###################### MultiSkip Residual MLP ######################
class StoreGlobalSkip(nn.Module):
    def forward(self, x):
        return (x, x)

class MultiSkipResidualBlock(nn.Module):
    def __init__(self, hidden_size, use_ln=False, activation_fn="relu", device='cpu'):
        super().__init__()
        self.block = ResidualBlock(hidden_size, use_ln=use_ln, activation_fn=activation_fn, device=device)
        
    def forward(self, x_tuple):
        x, global_skip = x_tuple
        out = self.block(x)
        out = out + global_skip
        return (out, global_skip)

class ExtractOutput(nn.Module):
    def forward(self, x_tuple):
        x, _ = x_tuple
        return x

class MultiSkipResidualMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        last_act=False,
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super().__init__()
        self.output_size = output_size
        act_fn_class = get_act_fn_clss(activation_fn)
        layers = []
        
        # Initial linear layer and optional layer norm.
        layers.append(layer_init(nn.Linear(input_size, hidden_size, device=device)))
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size, device=device))
        
        # Store the output as the global skip.
        layers.append(StoreGlobalSkip())
        
        # Append the sequence of multi-skip residual blocks.
        for _ in range(num_layers):
            layers.append(MultiSkipResidualBlock(hidden_size, use_ln=use_ln, activation_fn=activation_fn, device=device))
        
        # Extract the current output (discarding the global skip from the tuple).
        layers.append(ExtractOutput())
        
        # Final linear layer and optional layer norm/activation.
        layers.append(layer_init(nn.Linear(hidden_size, output_size, device=device)))
        if use_ln:
            layers.append(nn.LayerNorm(output_size, device=device))
        if last_act:
            layers.append(act_fn_class())
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)