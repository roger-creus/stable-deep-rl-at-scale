import torch
import numpy as np
import torch.nn as nn
from utils.utils import get_act_fn_clss, get_act_fn_functional
from torch.nn.utils import spectral_norm

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
        use_spectral_norm=False,
        activation_fn="relu",
        device='cpu',
        linear_clss=nn.Linear
    ):
        super(MLP, self).__init__()
        self.output_size = output_size
        
        mlp = []
        act_ = get_act_fn_clss(activation_fn)
        
        if num_layers == 1: hidden_size = output_size
        
        for i in range(num_layers):
            # Input layer
            if i == 0:
                if linear_clss.__name__ == 'NoisyLinear':
                    layer = linear_clss(input_size, hidden_size, device=device)
                else:
                    layer = layer_init(linear_clss(input_size, hidden_size, device=device))
                if use_spectral_norm:
                    layer = spectral_norm(layer)
                mlp.append(layer)
            # Output layer
            elif i == num_layers - 1:
                if linear_clss.__name__ == 'NoisyLinear':
                    layer = linear_clss(hidden_size, output_size, device=device)
                else:
                    layer = layer_init(linear_clss(hidden_size, output_size, device=device))
                if use_spectral_norm:
                    layer = spectral_norm(layer)
                mlp.append(layer)
            # Hidden layers
            else:
                if linear_clss.__name__ == 'NoisyLinear':
                    layer = linear_clss(hidden_size, hidden_size, device=device)
                else:
                    layer = layer_init(linear_clss(hidden_size, hidden_size, device=device))
                if use_spectral_norm:
                    layer = spectral_norm(layer)
                mlp.append(layer)
                
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
                if last_act and use_ln:
                    mlp.append(
                        nn.LayerNorm(output_size, device=device)
                    )
                elif last_act:
                    mlp.append(act_())
                        
                    
                   
            
        self.net = nn.Sequential(*mlp)
        
    def forward(self, x):
        return self.net(x)
        
class ResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        use_ln=False,
        use_spectral_norm=False,
        activation_fn="relu",
        device='cpu',
        linear_clss=nn.Linear
    ):
        super().__init__()
        self.use_ln = use_ln
        self.act_ = get_act_fn_functional(activation_fn)
        
        if linear_clss.__name__ == 'NoisyLinear':
            self.linear0 = linear_clss(hidden_size, hidden_size, device=device)
            self.linear1 = linear_clss(hidden_size, hidden_size, device=device)
        else:
            self.linear0 = layer_init(linear_clss(hidden_size, hidden_size, device=device))
            self.linear1 = layer_init(linear_clss(hidden_size, hidden_size, device=device))
        
        if use_spectral_norm:
            self.linear0 = spectral_norm(self.linear0)
            self.linear1 = spectral_norm(self.linear1)
            
        if self.use_ln:
            self.ln0 = nn.LayerNorm(hidden_size, device=device)
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
        use_spectral_norm=False,
        activation_fn="relu",
        device='cpu',
        linear_clss=nn.Linear
    ):
        super().__init__()
        self.output_size = output_size
        act_fn_class = get_act_fn_clss(activation_fn)
        layers = []
        
        if linear_clss.__name__ == 'NoisyLinear':
            layer = linear_clss(input_size, hidden_size, device=device)
        else:
            layer = layer_init(linear_clss(input_size, hidden_size, device=device))
        
        if use_spectral_norm:
            layer = spectral_norm(layer)
        layers.append(layer)
            
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size, device=device))
        
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_size, use_ln=use_ln, use_spectral_norm=use_spectral_norm, activation_fn=activation_fn, device=device, linear_clss=linear_clss))
        
        if linear_clss.__name__ == 'NoisyLinear':
            layer = linear_clss(hidden_size, output_size, device=device)
        else:
            layer = layer_init(linear_clss(hidden_size, output_size, device=device))
        
        if use_spectral_norm:
            layer = spectral_norm(layer)
        layers.append(layer)
            
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
    def __init__(self, hidden_size, use_ln=False, use_spectral_norm=False, activation_fn="relu", device='cpu', linear_clss=nn.Linear):
        super().__init__()
        self.block = ResidualBlock(hidden_size, use_ln=use_ln, use_spectral_norm=use_spectral_norm, activation_fn=activation_fn, device=device, linear_clss=linear_clss)
        
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
        use_spectral_norm=False,
        activation_fn="relu",
        device='cpu',
        linear_clss=nn.Linear
    ):
        super().__init__()
        self.output_size = output_size
        act_fn_class = get_act_fn_clss(activation_fn)
        layers = []
        
        # Initial linear layer and optional layer norm.
        if linear_clss.__name__ == 'NoisyLinear':
            layer = linear_clss(input_size, hidden_size, device=device)
        else:
            layer = layer_init(linear_clss(input_size, hidden_size, device=device))
        
        if use_spectral_norm:
            layer = spectral_norm(layer)
        layers.append(layer)
            
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size, device=device))
        
        # Store the output as the global skip.
        layers.append(StoreGlobalSkip())
        
        # Append the sequence of multi-skip residual blocks.
        for _ in range(num_layers):
            layers.append(MultiSkipResidualBlock(hidden_size, use_ln=use_ln, use_spectral_norm=use_spectral_norm, activation_fn=activation_fn, device=device, linear_clss=linear_clss))
        
        # Extract the current output (discarding the global skip from the tuple).
        layers.append(ExtractOutput())
        
        # Final linear layer and optional layer norm/activation.
        if linear_clss.__name__ == 'NoisyLinear':
            layer = linear_clss(hidden_size, output_size, device=device)
        else:
            layer = layer_init(linear_clss(hidden_size, output_size, device=device))
        
        if use_spectral_norm:
            layer = spectral_norm(layer)
        layers.append(layer)
            
        if use_ln:
            layers.append(nn.LayerNorm(output_size, device=device))
        if last_act:
            layers.append(act_fn_class())
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)