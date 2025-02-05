import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from models.encoder import AtariCNN, ImpalaCNN, ConvSequence
from models.mlp import MLP
from models.transformer import Transformer
from utils.utils import get_act_fn_clss, find_all_modules

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


######################################## PQN Agents ########################################
class PQNAgent(nn.Module):
    def __init__(
        self,
        envs,
        use_ln=True,
        cnn_type="atari",
        cnn_channels=[32, 64, 64],
        trunk_hidden_size=512,
        trunk_output_size=512,
        trunk_num_layers=1,
        activation_fn="relu",
        device=None
    ):
        super().__init__()
        
        self.use_ln = use_ln
        act_ = get_act_fn_clss(activation_fn)
        
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
        
        # compute output size of network
        with torch.no_grad():
            dummy = envs.single_observation_space.sample()
            dummy_torch = torch.as_tensor(dummy).unsqueeze(0).float().to(device)
            dummy = self.network(dummy_torch)
            dummy_shape = dummy.view(dummy.size(0), -1).shape[1]
            
        self.trunk = MLP(
            input_size=dummy_shape,
            hidden_size=trunk_hidden_size, # will not be used if num_layers=1
            output_size=trunk_output_size,
            num_layers=trunk_num_layers,
            use_ln=use_ln,
            last_act=False,
            activation_fn=activation_fn,
            device=device
        )
        
        self.q_func = nn.Sequential(
            act_(),
            layer_init(nn.Linear(trunk_output_size, envs.single_action_space.n, device=device), std=0.01),
        )

    def forward(self, x):
        hidden = self.get_representation(x)
        return self.get_Q(hidden)
    
    def get_representation(self, x, dead_neurons=False, per_layer=False):
        x = x / 255.0
        neurons = []
        count = 0
        clss_to_hook = nn.LayerNorm if self.use_ln else nn.Conv2d
        for module in self.network.cnn.children():
            x = module(x)
            if (dead_neurons or per_layer) and (isinstance(module, clss_to_hook) or isinstance(module, ConvSequence)):
                neurons.append(
                    (f'cnn_{count}', F.relu(x.clone()).detach())
                )
                count += 1
                
        count = 0
        clss_to_hook = nn.LayerNorm if self.use_ln else nn.Linear
        for module in self.trunk.net.children():
            x = module(x)
            if (dead_neurons or per_layer) and isinstance(module, clss_to_hook):
                neurons.append(
                    (f'mlp_{count}', F.relu(x.clone()).detach())
                )
                count += 1

        if per_layer:
            neurons_dict = {}
            for layer, ns in neurons:
                neurons_dict[layer] = ns
            return x, neurons_dict

        if dead_neurons:
            dead_neurons = self.calculate_dead_neurons(neurons)
            return x, dead_neurons
        
        return x
    
    def get_layer_shapes(self):
        layer_shapes = {}
        count = 0
        
        clss_to_hook = nn.LayerNorm if self.use_ln else nn.Conv2d
        for module in self.network.cnn.children():
            if isinstance(module, clss_to_hook) or isinstance(module, ConvSequence):
                if clss_to_hook == nn.LayerNorm:
                    layer_shapes[f'cnn_{count}'] = module.normalized_shape
                else:
                    layer_shapes[f'cnn_{count}'] = module.output_shape
                count += 1
                
        clss_to_hook = nn.LayerNorm if self.use_ln else nn.Linear
        count = 0
        for module in self.trunk.net.children():
            if isinstance(module, clss_to_hook):
                if clss_to_hook == nn.LayerNorm:
                    layer_shapes[f'mlp_{count}'] = module.normalized_shape
                else:
                    layer_shapes[f'mlp_{count}'] = module.output_shape
                count += 1
                    
        return layer_shapes

    def get_Q(self, hidden):
        return self.q_func(hidden)
        
    def get_max_value(self, x):
        return self.forward(x).max(1)[0]
    
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
    
    def adapt(self, td_errors, act_fn):
        assert act_fn == "heuristic_adarl", f"Only heuristic ADARL is manually adaptable, not {act_fn}"
        for module in find_all_modules(self, module_clss=get_act_fn_clss(act_fn)):
            module.adapt(td_errors)
    
    
######################################## PPO Agents ########################################
class SharedTrunkPPOAgent(nn.Module):
    def __init__(
        self,
        envs,
        use_ln=False,
        activation_fn="relu",
        cnn_type="atari",
        cnn_channels=[32, 64, 64],
        trunk_hidden_size=512,
        trunk_output_size=512,
        trunk_num_layers=1,
        device=None
    ):
        super().__init__()
        act_ = get_act_fn_clss(activation_fn)
        
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
        
        # compute output size of network
        with torch.no_grad():
            dummy = envs.single_observation_space.sample()
            dummy_torch = torch.as_tensor(dummy).unsqueeze(0).float().to(device)
            dummy = self.network(dummy_torch)
            dummy_shape = dummy.view(dummy.size(0), -1).shape[1]
            
        self.trunk = MLP(
            input_size=dummy_shape,
            hidden_size=trunk_hidden_size,
            output_size=trunk_output_size,
            num_layers=trunk_num_layers,
            activation_fn=activation_fn,
            use_ln=use_ln,
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

    def get_representation(self, x, dead_neurons=False):
        x = x / 255.0
        neurons = []
        clss_to_hook = nn.Conv2d
        for module in self.network.cnn.children():
            x = module(x)
            if (isinstance(module, clss_to_hook) or isinstance(module, ConvSequence)):
                neurons.append(
                    ('cnn', F.relu(x.clone()).detach())
                )
                
        clss_to_hook = nn.Linear
        for module in self.trunk.net.children():
            x = module(x)
            if isinstance(module, clss_to_hook):
                neurons.append(
                    ('mlp', F.relu(x.clone()).detach())
                )
                
        dead_neurons = self.calculate_dead_neurons(neurons)
        return x, dead_neurons
            
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
            total[layer] += torch.numel(mask)
            dead[layer] += (mask.sum().item())
        
        fraction_dead = {
            "cnn" : dead["cnn"]/total["cnn"] * 100,
            "mlp" : dead["mlp"]/total["mlp"] * 100
        }
        return fraction_dead


class DecoupledPPOAgent(nn.Module):
    def __init__(
        self,
        envs,
        use_ln=False,
        cnn_type="atari",
        cnn_channels=[32, 64, 64],
        activation_fn="relu",
        trunk_hidden_size=512,
        trunk_output_size=512,
        trunk_num_layers=1,
        device=None
    ):
        super().__init__()
        
        if cnn_type == "atari":
            network_clss = AtariCNN
        elif cnn_type == "impala":
            network_clss = ImpalaCNN
        else:
            raise NotImplementedError(f"Unknown cnn_type: {cnn_type}")
        
        # compute output size of network
        with torch.no_grad():
            dummy_enc = network_clss(
                cnn_channels=cnn_channels,
                use_ln=use_ln,
                device=device
            )
            dummy = envs.single_observation_space.sample()
            dummy_torch = torch.as_tensor(dummy).unsqueeze(0).float().to(device)
            dummy = dummy_enc(dummy_torch)
            dummy_shape = dummy.view(dummy.size(0), -1).shape[1]
            del dummy_enc
            del dummy
            del dummy_torch
            
        self.actor = nn.Sequential(
            network_clss(
                cnn_channels=cnn_channels,
                use_ln=use_ln,
                activation_fn=activation_fn,
                device=device
            ),
            MLP(
                input_size=dummy_shape,
                hidden_size=trunk_hidden_size,
                output_size=trunk_output_size,
                num_layers=trunk_num_layers,
                activation_fn=activation_fn,
                use_ln=use_ln,
                last_act=True,
                device=device
            ),
            layer_init(nn.Linear(trunk_output_size, envs.single_action_space.n, device=device), std=0.01)
        )
        
        self.critic = nn.Sequential(
            network_clss(
                cnn_channels=cnn_channels,
                activation_fn=activation_fn,
                use_ln=use_ln,
                device=device
            ),
            MLP(
                input_size=dummy_shape,
                hidden_size=trunk_hidden_size,
                output_size=trunk_output_size,
                num_layers=trunk_num_layers,
                use_ln=use_ln,
                activation_fn=activation_fn,
                last_act=True,
                device=device
            ),
            layer_init(nn.Linear(trunk_output_size, 1, device=device), std=1)
        )

    def get_value(self, x):
        return self.critic(x / 255.0)

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs / 255.0)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs / 255.0)
    
class TransformerPPOAgent(nn.Module):
    def __init__(
        self,
        envs,
        use_ln=False,
        activation_fn="relu",
        cnn_type="atari",
        cnn_channels=[32, 64, 64],
        trxl_dim=512,
        trxl_num_layers=1,
        trxl_num_heads=8,
        trxl_positional_encoding="absolute",
        max_episode_steps=27000,
        device="cuda"
    ):
        super().__init__()
        act_ = get_act_fn_clss(activation_fn)
        
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
        
        with torch.no_grad():
            dummy = envs.single_observation_space.sample()
            dummy_torch = torch.as_tensor(dummy).unsqueeze(0).float().to(device)
            dummy = self.network(dummy_torch)
            dummy_shape = dummy.view(dummy.size(0), -1).shape[1]
        
        self.trunk = MLP(
            input_size=dummy_shape,
            hidden_size=trxl_dim,
            output_size=trxl_dim,
            num_layers=1,
            activation_fn=activation_fn,
            use_ln=use_ln,
            last_act=True,
            device=device
        )
        
        self.trxl = Transformer(
            num_layers=trxl_num_layers,
            dim=trxl_dim,
            num_heads=trxl_num_heads,
            max_episode_steps=max_episode_steps,
            positional_encoding=trxl_positional_encoding,
            device=device
        )
        
        self.post_trxl = MLP(
            input_size=trxl_dim,
            hidden_size=trxl_dim,
            output_size=trxl_dim,
            num_layers=1,
            activation_fn=activation_fn,
            use_ln=use_ln,
            last_act=False,
            device=device
        )
        
        self.actor = nn.Sequential(
            act_(),
            layer_init(nn.Linear(trxl_dim, envs.single_action_space.n, device=device), std=0.01)
        )
        
        self.critic = nn.Sequential(
            act_(),
            layer_init(nn.Linear(trxl_dim, 1, device=device), std=1)
        )
    
    def get_value(self, x, memory, memory_mask, memory_indices):
        x = self.network(x / 255.0)
        x = self.trunk(x)
        x, _ = self.trxl(x, memory, memory_mask, memory_indices)
        x = self.post_trxl(x)
        return self.critic(x)
    
    def get_action_and_value(self, obs, memory, memory_mask, memory_indices, action=None):
        features = self.network(obs / 255.0)
        features = self.trunk(features)
        hidden, memory = self.trxl(features, memory, memory_mask, memory_indices)
        hidden = self.post_trxl(hidden)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), memory