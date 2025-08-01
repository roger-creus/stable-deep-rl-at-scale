# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer

from utils.utils import parse_mlp_width, parse_mlp_depth, get_grad_norms, get_optimizer
from models.encoder import AtariCNN, ImpalaCNN
from models.agent import MLP, ResidualMLP, MultiSkipResidualMLP

@dataclass
class Args:
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
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "DemonAttackNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    
    mlp_width: str = "small"
    """the width of the MLP"""
    mlp_depth: str = "small"
    """the depth of the MLP"""
    optimizer: str = "adam" # adam, kron
    """the optimizer to use"""
    use_ln: bool = False
    """whether to use layer normalization"""
    cnn_type: str = "atari" # atari, impala
    """the type of CNN to use"""
    mlp_type: str = "default" # default, residual, multiskip_residual
    """the type of MLP to use"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here
class QNetwork(nn.Module):
    def __init__(self, env, mlp_type, mlp_width, mlp_depth, use_ln, cnn_type, device):
        super().__init__()
        
        if cnn_type == "atari":
            self.cnn = AtariCNN(
                cnn_channels=[32, 64, 64],
                use_ln=use_ln,
                activation_fn="relu",
                device=device
            )
        elif cnn_type == "impala":
            self.cnn = ImpalaCNN(
                cnn_channels=[32, 64, 64],
                use_ln=use_ln,
                activation_fn="relu",
                device=device
            )
        else:
            raise ValueError(f"Invalid CNN type: {cnn_type}")
        
        if mlp_type == "default":
            mlp_clss = MLP
        elif mlp_type == "residual":
            mlp_clss = ResidualMLP
        elif mlp_type == "multiskip_residual":
            mlp_clss = MultiSkipResidualMLP
        else:
            raise ValueError(f"Invalid MLP type: {mlp_type}")
        
        with torch.no_grad():
            dummy = env.single_observation_space.sample()
            dummy_torch = torch.as_tensor(dummy).unsqueeze(0).float().to(device)
            dummy_features = self.cnn(dummy_torch)
            input_dim = dummy_features.view(dummy_features.size(0), -1).shape[1]
        
        self.trunk = mlp_clss(
            input_size=input_dim,
            hidden_size=mlp_width,
            output_size=512,
            num_layers=mlp_depth,
            use_ln=use_ln,
            device=device,
            last_act=True
        )
        
        self.q_head = nn.Linear(512, env.single_action_space.n)

    def forward(self, x):
        x = x / 255.0
        features = self.cnn(x)
        return self.q_head(self.trunk(features))


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"DQN_ENV:{args.env_id}_OPTIM:{args.optimizer}_MLP.TYPE:{args.mlp_type}_MLP.DEPTH:{args.mlp_depth}_MLP.WIDTH:{args.mlp_width}_LN:{args.use_ln}_CNN:{args.cnn_type}_SEED:{args.seed}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    layer_width = parse_mlp_width(args.mlp_width)
    num_layers = parse_mlp_depth(args.mlp_depth, args.mlp_type)

    q_network = QNetwork(envs, args.mlp_type, layer_width, num_layers, args.use_ln, args.cnn_type, device).to(device)
    
    optimizer_clss = get_optimizer(args.optimizer)
    if args.optimizer == "kron":
        args.learning_rate /= 3.0
    
    optimizer = optimizer_clss(q_network.parameters(), lr=args.learning_rate)
    print(f"Optimizer: {optimizer}")
    
    target_network = QNetwork(envs, args.mlp_type, layer_width, num_layers, args.use_ln, args.cnn_type, device).to(device)
    target_network.load_state_dict(q_network.state_dict())
    print(q_network)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    #writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    #writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    if args.track:
                        wandb.log({"charts/episodic_return": info["episode"]["r"], "charts/episodic_length": info["episode"]["l"]})
                        
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    #writer.add_scalar("losses/td_loss", loss, global_step)
                    #writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    #writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    
                    grad_norms = get_grad_norms(q_network)
                    grad_norms = {f"grad_norms/{k}": v for k, v in grad_norms.items()}
                    
                    if args.track:
                        wandb.log({
                            "losses/td_loss": loss,
                            "losses/q_values": old_val.mean().item(),
                            "charts/SPS": int(global_step / (time.time() - start_time)),
                            "grad_norms": grad_norms
                        })

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    envs.close()