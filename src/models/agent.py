import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from models.encoder import AtariCNN, ImpalaCNN, ConvSequence
from models.mlp import MLP, ResidualMLP, ResidualBlock, MultiSkipResidualMLP, MultiSkipResidualBlock, DenseNetMLP, DenseBlock
from utils.utils import get_act_fn_clss, get_act_fn_functional

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)    
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class BasePQNAgent(nn.Module):
    def __init__(
        self,
        envs,
        use_ln=True,
        use_spectral_norm=False,
        cnn_type="atari",
        mlp_type="default",
        cnn_channels=[32, 64, 64],
        trunk_hidden_size=512,
        trunk_output_size=512,
        trunk_num_layers=1,
        activation_fn="relu",
        device=None,
        use_l2_norm=False,
        weight_clip_value=None
    ):
        super().__init__()
        self.envs = envs
        self.use_ln = use_ln
        self.cnn_type = cnn_type
        self.mlp_type = mlp_type
        self.device = device
        self.act_fn_F = get_act_fn_functional(activation_fn)
        
        
        mlp_kwargs = {}
        if mlp_type == "default":
            mlp_clss = MLP
            mlp_kwargs["weight_clip_value"] = weight_clip_value
        elif mlp_type == "residual":
            mlp_clss = ResidualMLP
        elif mlp_type == "multiskip_residual":
            mlp_clss = MultiSkipResidualMLP
        elif mlp_type == "densenet":
            mlp_clss = DenseNetMLP
        else:
            raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")
        
        if cnn_type == "atari":
            self.network = AtariCNN(
                cnn_channels=cnn_channels,
                use_ln=use_ln,
                activation_fn=activation_fn,
                device=device,
                use_l2_norm=use_l2_norm
            )
        elif cnn_type == "impala":
            self.network = ImpalaCNN(
                cnn_channels=cnn_channels,
                use_ln=use_ln,
                activation_fn=activation_fn,
                device=device
            )
        else:
            raise NotImplementedError(f"Unknown cnn_type: {cnn_type}")
        
        with torch.no_grad():
            dummy = envs.single_observation_space.sample()
            dummy_torch = torch.as_tensor(dummy).unsqueeze(0).float().to(device)
            dummy_features = self.network(dummy_torch)
            dummy_shape = dummy_features.view(dummy_features.size(0), -1).shape[1]
        
        self.trunk = mlp_clss(
            input_size=dummy_shape,
            hidden_size=trunk_hidden_size,
            output_size=trunk_output_size,
            num_layers=trunk_num_layers,
            use_ln=use_ln,
            use_spectral_norm=use_spectral_norm,
            activation_fn=activation_fn,
            device=device,
            last_act=False,
            **mlp_kwargs
        )
        
        self.action_dim = envs.single_action_space.n

    # get the representation of the input x. pass layer by layer to get every layer's intermediate representation.
    def get_representation(self, x, dead_neurons=False, per_layer=False):
        x = x / 255.0
        neurons = []

        def process_modules(modules, x, prefix, hook_cls, additional_cls):
            count = 0
            collected = []
            for module in modules:
                if isinstance(module, nn.Sequential):
                    x, c, sub_collected = process_modules(module.children(), x, prefix, hook_cls, additional_cls)
                    count += c
                    collected.extend(sub_collected)
                else:
                    x = module(x)
                    if dead_neurons or per_layer:
                        if isinstance(module, hook_cls):
                            collected.append((f'{prefix}_{count}', self.act_fn_F(x.clone()).detach()))
                            count += 1
                        elif isinstance(module, additional_cls):
                            for sub_module in module.children():
                                if isinstance(sub_module, ResidualBlock):
                                    for sub_sub_module in sub_module.children():
                                        if isinstance(sub_sub_module, hook_cls):
                                            if isinstance(x, tuple):
                                                collected.append((f'{prefix}_{count}', self.act_fn_F(x[0].clone()).detach()))
                                            else:
                                                collected.append((f'{prefix}_{count}', self.act_fn_F(x.clone()).detach()))
                                            count += 1
                                else:
                                    if isinstance(sub_module, hook_cls):
                                        collected.append((f'{prefix}_{count}', self.act_fn_F(x.clone()).detach()))
                                        count += 1
            return x, count, collected

        cnn_hook = nn.LayerNorm if self.use_ln else nn.Conv2d
        cnn_additional = ConvSequence  
        x, _, cnn_neurons = process_modules(self.network.cnn.children(), x, 'cnn', cnn_hook, cnn_additional)
        neurons.extend(cnn_neurons)

        x = x.view(x.size(0), -1)

        mlp_hook = nn.LayerNorm if self.use_ln else nn.Linear
        mlp_additional = (ResidualBlock, MultiSkipResidualBlock)
        x, _, mlp_neurons = process_modules(self.trunk.net.children(), x, 'mlp', mlp_hook, mlp_additional)
        neurons.extend(mlp_neurons)

        if per_layer:
            neurons_dict = {layer: act for layer, act in neurons}
            return x, neurons_dict
        if dead_neurons:
            dead = self.calculate_dead_neurons(neurons)
            return x, dead
        return x

    def get_layer_shapes(self):
        layer_shapes = {}
        device = next(self.parameters()).device
        dummy_input = torch.rand(1, 4, 84, 84).to(device)

        def process_modules(modules, x, prefix, hook_cls, additional_cls):
            count = 0
            shapes = {}
            for module in modules:
                if isinstance(module, nn.Sequential):
                    x, c, sub_shapes = process_modules(module.children(), x, prefix, hook_cls, additional_cls)
                    count += c
                    shapes.update(sub_shapes)
                else:
                    with torch.no_grad():
                        if isinstance(module, MultiSkipResidualBlock) and not isinstance(x, tuple):
                            x = (x, x)
                        x = module(x)

                    if isinstance(module, hook_cls):
                        shapes[f'{prefix}_{count}'] = x[0].shape if isinstance(x, tuple) else x.shape
                        count += 1
                    elif isinstance(module, additional_cls):
                        for sub_module in module.children():
                            if isinstance(sub_module, ResidualBlock):
                                for sub_sub_module in sub_module.children():
                                    if isinstance(sub_sub_module, hook_cls):
                                        shapes[f'{prefix}_{count}'] = x[0].shape if isinstance(x, tuple) else x.shape
                                        count += 1
                            else:
                                if isinstance(sub_module, hook_cls):
                                    shapes[f'{prefix}_{count}'] = x.shape
                                    count += 1

            return x, count, shapes

        cnn_hook = nn.LayerNorm if self.use_ln else nn.Conv2d
        cnn_additional = ConvSequence
        x, _, cnn_shapes = process_modules(self.network.cnn.children(), dummy_input, 'cnn', cnn_hook, cnn_additional)
        layer_shapes.update(cnn_shapes)

        x = x.view(x.size(0), -1)

        mlp_hook = nn.LayerNorm if self.use_ln else nn.Linear
        mlp_additional = (ResidualBlock, MultiSkipResidualBlock, DenseBlock)
        x, _, mlp_shapes = process_modules(self.trunk.net.children(), x, 'mlp', mlp_hook, mlp_additional)
        layer_shapes.update(mlp_shapes)
        return layer_shapes

    def calculate_dead_neurons(self, neurons):
        total = {"cnn": 0, "mlp": 0}
        dead = {"cnn": 0, "mlp": 0}
        
        for layer, ns in neurons:
            score = ns.mean(dim=0)
            mask = score <= 0.0
            layer_name = layer.split("_")[0]
            total[layer_name] += torch.numel(mask)
            dead[layer_name] += mask.sum().item()
        
        fraction_dead = {
            "cnn": dead["cnn"] / total["cnn"] * 100 if total["cnn"] > 0 else 0,
            "mlp": dead["mlp"] / total["mlp"] * 100 if total["mlp"] > 0 else 0
        }
        return fraction_dead

# ============================================================================
# Standard PQN Agent
# ============================================================================

class PQNAgent(BasePQNAgent):
    def __init__(
        self,
        envs,
        use_ln=True,
        use_spectral_norm=False,
        cnn_type="atari",
        mlp_type="default",
        cnn_channels=[32, 64, 64],
        trunk_hidden_size=512,
        trunk_output_size=512,
        trunk_num_layers=1,
        activation_fn="relu",
        device=None,
        use_l2_norm=False,
        weight_clip_value=None
    ):
        super().__init__(
            envs,
            use_ln=use_ln,
            use_spectral_norm=use_spectral_norm,
            cnn_type=cnn_type,
            mlp_type=mlp_type,
            cnn_channels=cnn_channels,
            trunk_hidden_size=trunk_hidden_size,
            trunk_output_size=trunk_output_size,
            trunk_num_layers=trunk_num_layers,
            activation_fn=activation_fn,
            device=device,
            use_l2_norm=use_l2_norm,
            weight_clip_value=weight_clip_value
        )
        act_ = get_act_fn_clss(activation_fn)
        self.q_func = nn.Sequential(
            act_(),
            layer_init(nn.Linear(trunk_output_size, self.action_dim, device=device), std=0.01),
        )

    def forward(self, x):
        hidden = self.get_representation(x)
        return self.get_Q(hidden)
    
    def get_Q(self, hidden):
        return self.q_func(hidden)
        
    def get_max_value(self, x):
        return self.forward(x).max(1)[0]
    
    
######################################## PPO Agents ########################################
class SharedTrunkPPOAgent(nn.Module):
    def __init__(
        self,
        envs,
        use_ln=False,
        use_spectral_norm=False,
        activation_fn="relu",
        cnn_type="atari",
        mlp_type="default",
        cnn_channels=[32, 64, 64],
        trunk_hidden_size=512,
        trunk_output_size=512,
        trunk_num_layers=1,
        device=None
    ):
        super().__init__()
        self.use_ln = use_ln
        act_ = get_act_fn_clss(activation_fn)
        self.act_fn_F = get_act_fn_functional(activation_fn)
        
        if cnn_type == "atari":
            self.network = AtariCNN(
                cnn_channels=cnn_channels,
                use_ln=use_ln,
                activation_fn=activation_fn,
                device=device
            )
        elif cnn_type == "impala":
            self.network = ImpalaCNN(
                cnn_channels=cnn_channels,
                use_ln=use_ln,
                activation_fn=activation_fn,
                device=device
            )
        else:
            raise NotImplementedError(f"Unknown cnn_type: {cnn_type}")
        
        if mlp_type == "default":
            mlp_clss = MLP
        elif mlp_type == "residual":
            mlp_clss = ResidualMLP
        elif mlp_type == "multiskip_residual":
            mlp_clss = MultiSkipResidualMLP
        elif mlp_type == "densenet":
            mlp_clss = DenseNetMLP
        else:
            raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")
        
        with torch.no_grad():
            dummy = envs.single_observation_space.sample()
            dummy_torch = torch.as_tensor(dummy).unsqueeze(0).float().to(device)
            dummy = self.network(dummy_torch)
            dummy_shape = dummy.view(dummy.size(0), -1).shape[1]
            
        self.trunk = mlp_clss(
            input_size=dummy_shape,
            hidden_size=trunk_hidden_size,
            output_size=trunk_output_size,
            num_layers=trunk_num_layers,
            activation_fn=activation_fn,
            use_ln=use_ln,
            use_spectral_norm=use_spectral_norm,
            last_act=False,
            device=device
        )
        
        self.actor = nn.Sequential(
            act_(),
            layer_init(nn.Linear(trunk_output_size, envs.single_action_space.n, device=device), std=0.01)    
        )
        
        self.critic = nn.Sequential(
            act_(),
            layer_init(nn.Linear(trunk_output_size, 1, device=device), std=1)
        )
            
    def get_value(self, x):
        features = self.network(x / 255.0)
        hidden = self.trunk(features)
        value = self.critic(hidden)
        return value
    
    def get_action_and_value(self, obs, action=None):
        features = self.network(obs / 255.0)
        hidden = self.trunk(features)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def calculate_dead_neurons(self, neurons):
        total = {
            "cnn" : 0,
            "mlp" : 0
        }
        dead = {
            "cnn" : 0,
            "mlp" : 0
        }
        
        for layer, ns in neurons:
            score = ns.mean(dim=0)
            mask = score <= 0.0
            layer_name = layer.split("_")[0]
            total[layer_name] += torch.numel(mask)
            dead[layer_name] += (mask.sum().item())
        
        fraction_dead = {
            "cnn" : dead["cnn"]/total["cnn"] * 100,
            "mlp" : dead["mlp"]/total["mlp"] * 100
        }
        return fraction_dead
    
    def get_representation(self, x, dead_neurons=False, per_layer=False):
        x = x / 255.0
        neurons = []

        def process_modules(modules, x, prefix, hook_cls, additional_cls):
            count = 0
            collected = []
            for module in modules:
                if isinstance(module, nn.Sequential):
                    x, c, sub_collected = process_modules(module.children(), x, prefix, hook_cls, additional_cls)
                    count += c
                    collected.extend(sub_collected)
                else:
                    x = module(x)
                    if dead_neurons or per_layer:
                        if isinstance(module, hook_cls):
                            collected.append((f'{prefix}_{count}', self.act_fn_F(x.clone()).detach()))
                            count += 1
                        elif isinstance(module, additional_cls):
                            for sub_module in module.children():
                                if isinstance(sub_module, ResidualBlock):
                                    for sub_sub_module in sub_module.children():
                                        if isinstance(sub_sub_module, hook_cls):
                                            if isinstance(x, tuple):
                                                collected.append((f'{prefix}_{count}', self.act_fn_F(x[0].clone()).detach()))
                                            else:
                                                collected.append((f'{prefix}_{count}', self.act_fn_F(x.clone()).detach()))
                                            count += 1
                                else:
                                    if isinstance(sub_module, hook_cls):
                                        collected.append((f'{prefix}_{count}', self.act_fn_F(x.clone()).detach()))
                                        count += 1
            return x, count, collected

        cnn_hook = nn.LayerNorm if self.use_ln else nn.Conv2d
        cnn_additional = ConvSequence  
        x, _, cnn_neurons = process_modules(self.network.cnn.children(), x, 'cnn', cnn_hook, cnn_additional)
        neurons.extend(cnn_neurons)

        x = x.view(x.size(0), -1)

        mlp_hook = nn.LayerNorm if self.use_ln else nn.Linear
        mlp_additional = (ResidualBlock, MultiSkipResidualBlock)
        x, _, mlp_neurons = process_modules(self.trunk.net.children(), x, 'mlp', mlp_hook, mlp_additional)
        neurons.extend(mlp_neurons)

        if per_layer:
            neurons_dict = {layer: act for layer, act in neurons}
            return x, neurons_dict
        if dead_neurons:
            dead = self.calculate_dead_neurons(neurons)
            return x, dead
        return x

    def get_layer_shapes(self):
        layer_shapes = {}
        device = next(self.parameters()).device
        dummy_input = torch.rand(1, 4, 84, 84).to(device)

        def process_modules(modules, x, prefix, hook_cls, additional_cls):
            count = 0
            shapes = {}
            for module in modules:
                if isinstance(module, nn.Sequential):
                    x, c, sub_shapes = process_modules(module.children(), x, prefix, hook_cls, additional_cls)
                    count += c
                    shapes.update(sub_shapes)
                else:
                    with torch.no_grad():
                        if isinstance(module, MultiSkipResidualBlock) and not isinstance(x, tuple):
                            x = (x, x)
                        x = module(x)

                    if isinstance(module, hook_cls):
                        shapes[f'{prefix}_{count}'] = x[0].shape if isinstance(x, tuple) else x.shape
                        count += 1
                    elif isinstance(module, additional_cls):
                        for sub_module in module.children():
                            if isinstance(sub_module, ResidualBlock):
                                for sub_sub_module in sub_module.children():
                                    if isinstance(sub_sub_module, hook_cls):
                                        shapes[f'{prefix}_{count}'] = x[0].shape if isinstance(x, tuple) else x.shape
                                        count += 1
                            else:
                                if isinstance(sub_module, hook_cls):
                                    shapes[f'{prefix}_{count}'] = x.shape
                                    count += 1

            return x, count, shapes

        cnn_hook = nn.LayerNorm if self.use_ln else nn.Conv2d
        cnn_additional = ConvSequence
        x, _, cnn_shapes = process_modules(self.network.cnn.children(), dummy_input, 'cnn', cnn_hook, cnn_additional)
        layer_shapes.update(cnn_shapes)

        x = x.view(x.size(0), -1)

        mlp_hook = nn.LayerNorm if self.use_ln else nn.Linear
        mlp_additional = (ResidualBlock, MultiSkipResidualBlock)
        x, _, mlp_shapes = process_modules(self.trunk.net.children(), x, 'mlp', mlp_hook, mlp_additional)
        layer_shapes.update(mlp_shapes)

        return layer_shapes