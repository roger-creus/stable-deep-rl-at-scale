import random
import time
import os
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils.craftax_utils import make_craftax_env, write_row_csv, make_training_csv_craftax_classic, make_training_csv_craftax
from utils.utils import parse_mlp_width, parse_mlp_depth, get_grad_norms, get_optimizer
from models.agent import MLP, ResidualMLP, MultiSkipResidualMLP

@dataclass
class PQN_Craftax_Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "pqn_lstm_craftax"
    """the wandb's project name"""
    wandb_entity: str = "rogercreus"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    env_id: str = "Craftax"
    """the id of the environment"""
    total_timesteps: int = 1_000_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.00025
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel game environments"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    q_lambda: float = 0.65
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-btches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    log_every: int = 10
    """how often to log the training progress (every how many episodes)"""

    optimizer: str = "radam" # radam, kron
    """the optimizer to use"""
    mlp_width: str = "small"
    """the width of the MLP"""
    mlp_depth: str = "small"
    """the depth of the MLP"""
    mlp_type: str = "default"
    """the type of MLP"""
    use_ln: bool = True
    """whether to use layer normalization"""
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.005
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PQN_Craftax_Agent(nn.Module):
    def __init__(self, envs, mlp_type, mlp_width, mlp_depth, use_ln, device):
        super().__init__()
        
        obs_space = envs.single_observation_space
        
        if mlp_type == "default":
            mlp_clss = MLP
        elif mlp_type == "residual":
            mlp_clss = ResidualMLP
        elif mlp_type == "multiskip_residual":
            mlp_clss = MultiSkipResidualMLP
        else:
            raise ValueError(f"Invalid MLP type: {mlp_type}")
        
        self.network = mlp_clss(
            input_size=np.array(obs_space.shape).prod(),
            hidden_size=mlp_width,
            output_size=512,
            num_layers=mlp_depth,
            use_ln=use_ln,
            device=device,
            last_act=True,
        )
     
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        
        self.post_lstm = nn.Sequential(
            layer_init(nn.Linear(128, 128)),
            nn.LayerNorm(128),
            nn.ReLU(),
        )   
        
        self.q_func = nn.Sequential(
            layer_init(nn.Linear(128, envs.single_action_space.n))
        )

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d.float()).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d.float()).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        new_hidden = self.post_lstm(new_hidden)
        return new_hidden, lstm_state

    def forward(self, x, lstm_state, done):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        qvals = self.q_func(hidden)
        return qvals, lstm_state

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = tyro.cli(PQN_Craftax_Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = args.exp_name = f"algo:PQNLSTM_ENV:{args.env_id}_OPTIM:{args.optimizer}_MLP.TYPE:{args.mlp_type}_MLP.DEPTH:{args.mlp_depth}_MLP.WIDTH:{args.mlp_width}_LN:{args.use_ln}_SEED:{args.seed}"
    
    ################## Logging setup ##################
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if args.env_id == "Craftax-Classic":
        training_csv_writer = make_training_csv_craftax_classic(f"./runs/{run_name}")
    elif args.env_id == "Craftax":
        training_csv_writer = make_training_csv_craftax(f"./runs/{run_name}")
    else:
        raise ValueError(f"Unknown environment: {args.env_id}")
    #################################################

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"using device {device}")

    # env setup
    envs = make_craftax_env(
        env_id=args.env_id,
        num_envs=args.num_envs,
        device=device,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    layer_width = parse_mlp_width(args.mlp_width)
    num_layers = parse_mlp_depth(args.mlp_depth, args.mlp_type)

    # agent setup
    q_network = PQN_Craftax_Agent(envs, args.mlp_type, layer_width, num_layers, args.use_ln, device).to(device)
    print("-------------")
    print(q_network)
    
    optimizer_cls = get_optimizer(args.optimizer)
    if args.optimizer == "kron":
        optimizer = optimizer_cls(q_network.parameters(), lr=args.learning_rate / 1.25)
    else:
        optimizer = optimizer_cls(q_network.parameters(), lr=args.learning_rate)

    # ALGO Logic: Storage setup
    obs_shape = envs.single_observation_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    print("Starting the training process...")
    start_time = time.time()
    next_done = torch.zeros(args.num_envs).to(device)
    episode_count = 0
    
    next_lstm_state = (
        torch.zeros(1, args.num_envs, q_network.lstm.hidden_size).to(device),
        torch.zeros(1, args.num_envs, q_network.lstm.hidden_size).to(device),
    )

    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,)).to(device)
            
            with torch.no_grad():
                q_values, next_lstm_state = q_network(next_obs, next_lstm_state, next_done)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()

            explore = (torch.rand((args.num_envs,)).to(device) < epsilon)
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations)
            rewards[step] = reward.view(-1)

            for idx in range(args.num_envs):
                if next_done[idx]:
                    if episode_count % args.log_every == 0:
                        print(f"global_step={global_step}, episodic_return={infos['r'][idx]}, episodic_length={infos['l'][idx]}")
                        writer.add_scalar("charts/episodic_return", infos["r"][idx], global_step)
                        writer.add_scalar("charts/episodic_length", infos["l"][idx], global_step)
                        
                        # log craftax achievements
                        achievement_logs = {k:v[idx].item() for k,v in infos.items() if 'Achievements' in k}
                        row_to_write = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                        int(global_step / (time.time() - start_time)),
                                        global_step,
                                        infos["r"][idx],
                                        infos["l"][idx]] + list(achievement_logs.values())
                        write_row_csv(training_csv_writer, row_to_write)
                        for k, v in achievement_logs.items():
                            writer.add_scalar(f"Achievements/{k.split('/')[-1]}", v, global_step)
                    episode_count += 1

        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_value, _ = torch.max(q_network(next_obs, next_lstm_state, next_done)[0], dim=-1)
                    nextnonterminal = 1.0 - next_done.float()
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                    returns[t] = rewards[t] + args.gamma * (
                        args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value
                    ) * nextnonterminal

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_dones = dones.reshape(-1)

        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        # Optimizing the Q-network
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                old_val, _ = q_network(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds]
                )
                old_val = old_val.gather(1, b_actions[mb_inds].unsqueeze(-1).long()).squeeze()

                loss = F.mse_loss(b_returns[mb_inds], old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("losses/td_loss", loss, global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/epsilon", epsilon, global_step)

    torch.save(q_network.state_dict(), f"runs/{run_name}/agent_{iteration}.pt")
    writer.close()