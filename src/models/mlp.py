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
            if i == 0:
                if linear_clss.__name__ == 'NoisyLinear':
                    layer = linear_clss(input_size, hidden_size, device=device)
                else:
                    layer = layer_init(linear_clss(input_size, hidden_size, device=device))
                if use_spectral_norm:
                    layer = spectral_norm(layer)
                mlp.append(layer)
            elif i == num_layers - 1:
                if linear_clss.__name__ == 'NoisyLinear':
                    layer = linear_clss(hidden_size, output_size, device=device)
                else:
                    layer = layer_init(linear_clss(hidden_size, output_size, device=device))
                if use_spectral_norm:
                    layer = spectral_norm(layer)
                mlp.append(layer)
            else:
                if linear_clss.__name__ == 'NoisyLinear':
                    layer = linear_clss(hidden_size, hidden_size, device=device)
                else:
                    layer = layer_init(linear_clss(hidden_size, hidden_size, device=device))
                if use_spectral_norm:
                    layer = spectral_norm(layer)
                mlp.append(layer)
                
            if i < num_layers - 1:
                if use_ln:
                    mlp.append(
                        nn.LayerNorm(hidden_size, device=device)
                    )
                
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
        
        if linear_clss.__name__ == 'NoisyLinear':
            layer = linear_clss(input_size, hidden_size, device=device)
        else:
            layer = layer_init(linear_clss(input_size, hidden_size, device=device))
        
        if use_spectral_norm:
            layer = spectral_norm(layer)
        layers.append(layer)
            
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size, device=device))
        
        layers.append(StoreGlobalSkip())
        
        for _ in range(num_layers):
            layers.append(MultiSkipResidualBlock(hidden_size, use_ln=use_ln, use_spectral_norm=use_spectral_norm, activation_fn=activation_fn, device=device, linear_clss=linear_clss))
        
        layers.append(ExtractOutput())
        
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

###################### DenseNet MLP ######################
class DenseBlock(nn.Module):
    def __init__(
        self,
        input_size,
        growth_rate,
        use_ln=False,
        use_spectral_norm=False,
        activation_fn="relu",
        device='cpu',
        linear_clss=nn.Linear
    ):
        super().__init__()
        self.use_ln = use_ln
        self.act_fn = get_act_fn_clss(activation_fn)()
        
        if linear_clss.__name__ == 'NoisyLinear':
            self.linear = linear_clss(input_size, growth_rate, device=device)
        else:
            self.linear = layer_init(linear_clss(input_size, growth_rate, device=device))
        
        if use_spectral_norm:
            self.linear = spectral_norm(self.linear)
            
        if self.use_ln:
            self.ln = nn.LayerNorm(growth_rate, device=device)

    def forward(self, x):
        new_features = self.act_fn(x)
        new_features = self.linear(new_features)
        if self.use_ln:
            new_features = self.ln(new_features)
        return torch.cat([x, new_features], dim=1)

class DenseNetMLP(nn.Module):
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
        
        growth_rate = hidden_size // max(1, num_layers)
        if growth_rate == 0:
            growth_rate = hidden_size
        
        if linear_clss.__name__ == 'NoisyLinear':
            layer = linear_clss(input_size, hidden_size, device=device)
        else:
            layer = layer_init(linear_clss(input_size, hidden_size, device=device))
        
        if use_spectral_norm:
            layer = spectral_norm(layer)
        layers.append(layer)
            
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size, device=device))
        
        layers.append(act_fn_class())
        
        current_size = hidden_size
        for _ in range(num_layers):
            block = DenseBlock(
                input_size=current_size,
                growth_rate=growth_rate,
                use_ln=use_ln,
                use_spectral_norm=use_spectral_norm,
                activation_fn=activation_fn,
                device=device,
                linear_clss=linear_clss
            )
            layers.append(block)
            current_size += growth_rate
        
        if linear_clss.__name__ == 'NoisyLinear':
            layer = linear_clss(current_size, output_size, device=device)
        else:
            layer = layer_init(linear_clss(current_size, output_size, device=device))
        
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
    
    
### SUPER HADAMAX MLP ###
import torch.nn.functional as F

class SEBlock1D(nn.Module):
    def __init__(self, hidden_size, reduction=16, activation_fn="gelu", device='cpu'):
        super().__init__()
        reduced_size = max(1, hidden_size // reduction)
        self.fc1 = nn.Linear(hidden_size, reduced_size, device=device)
        self.act = get_act_fn_clss(activation_fn)()
        self.fc2 = nn.Linear(reduced_size, hidden_size, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.fc2(self.act(self.fc1(x)))
        scale = self.sigmoid(scale)
        return x * scale
    
class LearnableHadaFusionMLPBlock(nn.Module):
    def __init__(self, hidden_size, num_paths=4, use_ln=True,
                 activation_fn='gelu', device='cpu', linear_clss=nn.Linear,
                 se_reduction=16):
        super().__init__()
        self.paths = nn.ModuleList([
            layer_init(linear_clss(hidden_size, hidden_size, device=device))
            for _ in range(num_paths)
        ])
        self.lns = nn.ModuleList([
            nn.LayerNorm(hidden_size, device=device) if use_ln else nn.Identity()
            for _ in range(num_paths)
        ])
        self.fusion_weights = nn.Parameter(torch.ones(num_paths, device=device))
        self.mix_proj = layer_init(linear_clss(hidden_size, hidden_size, device=device))
        self.se = SEBlock1D(hidden_size, reduction=se_reduction, activation_fn=activation_fn, device=device)
        self.act = get_act_fn_functional(activation_fn)
        self.res_proj = linear_clss(hidden_size, hidden_size, device=device)

    def forward(self, x):
        outputs = [self.act(ln(fc(x))) for fc, ln in zip(self.paths, self.lns)]
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * o for w, o in zip(weights, outputs))
        mixed = self.mix_proj(fused)
        scaled = self.se(mixed)

        # Residual connection (with projection if needed)
        res = self.res_proj(x) if x.shape[-1] != scaled.shape[-1] else x
        return scaled + res

class SuperHadaMaxMLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 last_act=False,
                 use_ln=False,
                 activation_fn='relu',
                 use_spectral_norm=False,
                 linear_clss=nn.Linear,
                 device='cpu'
                 ):
        super().__init__()
        layers = []
        layers.append(layer_init(linear_clss(input_size, hidden_size, device=device)))
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size, device=device))
        layers.append(get_act_fn_clss(activation_fn)())
        for _ in range(num_layers):
            layers.append(LearnableHadaFusionMLPBlock(hidden_size, num_paths=4,
                                                      use_ln=use_ln,
                                                      activation_fn=activation_fn,
                                                      device=device,
                                                      linear_clss=linear_clss))
        layers.append(layer_init(linear_clss(hidden_size, output_size, device=device)))
        if use_ln:
            layers.append(nn.LayerNorm(output_size, device=device))
        if last_act:
            layers.append(get_act_fn_clss(activation_fn)())
        self.net = nn.Sequential(*layers)
        self.output_size = output_size

    def forward(self, x):
        return self.net(x)
