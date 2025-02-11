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
        self.use_ln = use_ln
        act_fn_class = get_act_fn_clss(activation_fn)
        self.activation = act_fn_class()
        
        self.initial = layer_init(nn.Linear(input_size, hidden_size, device=device))
        if use_ln:
            self.initial_ln = nn.LayerNorm(hidden_size, device=device)
        
        self.num_blocks = num_layers
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, use_ln=use_ln, activation_fn=activation_fn, device=device)
            for _ in range(num_layers)
        ])
        
        self.final = layer_init(nn.Linear(hidden_size, output_size, device=device))
        if use_ln:
            self.final_ln = nn.LayerNorm(output_size, device=device)
            
        self.last_act = act_fn_class() if last_act else None
        
    def forward(self, x):
        x = self.initial(x)
        if self.use_ln:
            x = self.initial_ln(x)
        global_skip = x
        
        for block in self.blocks:
            x = block(x)
            x = x + global_skip
            
        x = self.final(x)
        if self.use_ln:
            x = self.final_ln(x)
        if self.last_act is not None:
            x = self.last_act(x)
        return x

####################### DEEP RESIDUAL MLP #######################
class DenseResidualMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        activation_fn='relu',
        use_ln=False,
        device='cpu'
    ):
        super().__init__()
        act_ = get_act_fn_clss(activation_fn)
        self.activation = act_()
        self.num_layers = num_layers
        self.use_ln = use_ln

        self.layers = nn.ModuleList()
        if use_ln:
            self.norms = nn.ModuleList()

        current_dim = input_size
        for i in range(num_layers):
            layer = nn.Linear(current_dim, hidden_size, device=device)
            self.layers.append(layer)
            if use_ln:
                self.norms.append(nn.LayerNorm(hidden_size, device=device))
            current_dim += hidden_size
        self.final_proj = nn.Linear(current_dim, output_size, device=device)
        
    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.layers):
            x_cat = torch.cat(outputs, dim=1)
            out = layer(x_cat)
            if self.use_ln:
                out = self.norms[i](out)
            out = self.activation(out)
            outputs.append(out)
        x_cat = torch.cat(outputs, dim=1)
        return self.final_proj(x_cat)
