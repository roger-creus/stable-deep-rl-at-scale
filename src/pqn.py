import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import os
import random
import time
from collections import deque
import envpool

# import gymnasium as gym
import gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.categorical import Distribution

Distribution.set_default_validate_args(False)
torch.set_float32_matmul_precision("high")

from utils.compute_hns import _compute_hns
from utils.utils import parse_network_size
from utils.args import PQNArgs
from utils.compute_churn import compute_representation_and_q_churn
from utils.wrappers import RecordEpisodeStatistics
from models.agent import PQNAgent

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def lambda_returns(next_obs, next_done, container):
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)
    next_value = get_max_value(next_obs)
    returns = []
    returns.append(rewards[-1] + args.gamma * next_value * (~next_done).float())
    i = 1
    for t in range(args.num_steps - 2, -1, -1):
        next_value = vals_unbind[t+1]
        returns.append(
            rewards[t] + args.gamma * (
                args.q_lambda * returns[i-1] + (1 - args.q_lambda) * next_value * nextnonterminals[t+1]
            )
        )
        i+=1
            
    returns = container["returns"] = torch.stack(list(reversed(returns)))
    return container

def rollout(obs, done, avg_returns=[]):
    ts = []
    for step in range(args.num_steps):
        torch.compiler.cudagraph_mark_step_begin()
        q_values = policy(obs)
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,), device=device)
        max_actions = torch.argmax(q_values, dim=1)
        explore = (torch.rand((args.num_envs,), device=device) < epsilon)
        action = torch.where(explore, random_actions, max_actions)

        next_obs_np, reward, next_done, info = envs.step(action.cpu().numpy())
        next_obs = torch.as_tensor(next_obs_np)
        reward = torch.as_tensor(reward)
        next_done = torch.as_tensor(next_done)

        idx = next_done
        if idx.any():
            idx = idx & torch.as_tensor(info["lives"] == 0, device=next_done.device, dtype=torch.bool)
            if idx.any():
                r = torch.as_tensor(info["r"])
                avg_returns.extend(r[idx])

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                dones=done,
                vals=q_values[torch.arange(args.num_envs), max_actions].flatten(),
                actions=action,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )

        obs = next_obs = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container

def update(obs, actions, returns):
    optimizer.zero_grad()
    old_representation = old_agent.get_representation(obs)
    old_val = old_agent.get_Q(old_representation)
    old_val_gathered = old_val.gather(1, actions.unsqueeze(1)).squeeze(1)
    q_loss = F.mse_loss(returns, old_val_gathered)
    q_loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()
    
    return q_loss.detach(), gn

update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "returns"],
    out_keys=["td_loss", "var_loss", "gn"],
)

if __name__ == "__main__":
    args = tyro.cli(PQNArgs)

    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"PQN_CNN:{args.cnn_type}_SIZE:{args.network_size}_NENVS:{args.num_envs}_NSTEPS:{args.num_steps}_NEPOCHS:{args.update_epochs}_NMBS:{args.num_minibatches}_QLambda:{args.q_lambda}"
    args.run_name = run_name

    wandb.init(
        project="scaling_ppo",
        name=run_name + f"_{args.env_id}_{int(time.time())}_s{args.seed}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    ####### Agent #######
    cnn_channels, trunk_hidden_size, trunk_num_layers = parse_network_size(args.network_size)
    agent_cfg = {
        "envs": envs,
        "use_ln": args.use_ln,
        "cnn_type": args.cnn_type,
        "cnn_channels": cnn_channels,
        "trunk_hidden_size": trunk_hidden_size,
        "trunk_num_layers": trunk_num_layers,
        "device": device,
    }
    agent = PQNAgent(**agent_cfg)
    old_agent = PQNAgent(**agent_cfg)
    print(agent)
    
    # Make a version of agent with detached params
    agent_inference = PQNAgent(**agent_cfg)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    optimizer = optim.RAdam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.forward
    get_max_value = agent_inference.get_max_value

    # Compile policy
    if args.compile:
        mode = "reduce-overhead" if not args.cudagraphs else None
        policy = torch.compile(policy, mode=mode)
        lambda_returns = torch.compile(lambda_returns, fullgraph=True, mode=mode)
        update = torch.compile(update, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        #lambda_returns = CudaGraphModule(lambda_returns)
        update = CudaGraphModule(update)

    avg_returns = deque(maxlen=20)
    global_step = 0
    container_local = None
    next_obs = torch.tensor(envs.reset(), device=device, dtype=torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    max_ep_ret = -float("inf")
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    desc = ""
    global_step_burnin = None
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(next_obs, next_done, avg_returns=avg_returns)
        global_step += container.numel()
        
        torch.compiler.cudagraph_mark_step_begin()
        container = lambda_returns(next_obs, next_done, container)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        
        old_agent.load_state_dict(agent.state_dict())
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]
                torch.compiler.cudagraph_mark_step_begin()
                out = update(container_local, tensordict_out=tensordict.TensorDict())
    
        with torch.no_grad():
            churn_stats = compute_representation_and_q_churn(agent, old_agent, container_flat["obs"])

        if global_step_burnin is not None and iteration % 10 == 0:
            cur_time = time.time()
            speed = (global_step - global_step_burnin) / (cur_time - start_time)
            global_step_burnin = global_step
            start_time = cur_time

            r = container["rewards"].mean()
            r_max = container["rewards"].max()
            avg_returns_t = torch.tensor(avg_returns).mean()

            with torch.no_grad():
                ep_return = np.array(avg_returns).mean()
                logs = {
                    "episode_return": ep_return,
                    "hns": _compute_hns(args.env_id, ep_return),
                    "returns": container["returns"].mean(),
                    "vals": container["vals"].mean(),
                    "td_loss": out["td_loss"],
                    "gn": out["gn"].mean(),
                    "epsilon": linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step),
                }
                logs = {**logs, **churn_stats}

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"speed: {speed: 4.1f} sps, "
                f"reward avg: {r :4.2f}, "
                f"reward max: {r_max:4.2f}, "
                f"returns: {avg_returns_t: 4.2f},"
                f"lr: {lr: 4.2f}"
            )

            wandb.log(
                {"global_step": global_step, "speed": speed, "episode_return": avg_returns_t, "r": r, "r_max": r_max, "lr": lr, **logs}, step=global_step
            )

    envs.close()