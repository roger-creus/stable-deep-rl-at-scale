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
from torch.distributions.categorical import Categorical
from utils.craftax_utils import make_craftax_env, write_row_csv, make_training_csv_craftax_classic, make_training_csv_craftax
from utils.utils import parse_mlp_width, parse_mlp_depth, get_optimizer
from models.agent import MLP, ResidualMLP, MultiSkipResidualMLP

@dataclass
class PPO_Craftax_Args:
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
    wandb_project_name: str = "craftax-classic"
    """the wandb's project name"""
    wandb_entity: str = "rogercreus"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    env_id: str = "Craftax-Classic"
    """the id of the environment"""
    total_timesteps: int = 1_000_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.00025
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    log_every: int = 10
    """how often to log the training progress (every how many episodes)"""
    
    optimizer: str = "adam" # adam or kron
    """the optimizer to use"""
    mlp_width: str = "small"
    """the width of the MLP"""
    mlp_depth: str = "small"
    """the depth of the MLP"""
    mlp_type: str = "default"
    """the type of MLP"""
    use_ln: bool = False
    """whether to use layer normalization"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO_Craftax_Agent(nn.Module):
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
            activation_fn="tanh",
        )
     
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        
        self.post_lstm = nn.Sequential(
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )   
        
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=1.0)

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

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x,  lstm_state, done, action=None):
        hidden, new_lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), new_lstm_state

if __name__ == "__main__":
    args = tyro.cli(PPO_Craftax_Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = args.exp_name = f"algo:PPO_ENV:{args.env_id}_OPTIM:{args.optimizer}_MLP.TYPE:{args.mlp_type}_MLP.DEPTH:{args.mlp_depth}_MLP.WIDTH:{args.mlp_width}_SEED:{args.seed}"
    
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
    agent = PPO_Craftax_Agent(envs, args.mlp_type, layer_width, num_layers, args.use_ln, device).to(device)
    print("-------------")
    print(agent)
    
    optimizer_cls = get_optimizer(args.optimizer)
    if args.optimizer == "kron":
        optimizer = optimizer_cls(agent.parameters(), lr=args.learning_rate / 2.5)
    else:
        optimizer = optimizer_cls(agent.parameters(), lr=args.learning_rate)

    # ALGO Logic: Storage setup
    obs_shape = envs.single_observation_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
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
        torch.zeros(1, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(1, args.num_envs, agent.lstm.hidden_size).to(device),
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
            
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

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

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    torch.save(agent.state_dict(), f"runs/{run_name}/agent_{iteration}.pt")
    writer.close()