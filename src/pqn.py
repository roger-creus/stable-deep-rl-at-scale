import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["WANDB__SERVICE_WAIT"] = "300"

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
from utils.compute_churn import compute_representation_and_q_churn, compute_ranks_from_features, compute_visitation_distribution
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
                args.q_lambda * returns[i-1] + (1 - args.q_lambda) * next_value
            ) * nextnonterminals[t+1]
        )
        i+=1
            
    returns = container["returns"] = torch.stack(list(reversed(returns)))
    return container

def rollout(obs, done, avg_returns=[]):
    ts = []
    for step in range(args.num_steps):
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
    old_representation = agent.get_representation(obs)
    old_val = agent.get_Q(old_representation)
    old_val_gathered = old_val.gather(1, actions.unsqueeze(1)).squeeze(1)
    q_loss = F.mse_loss(old_val_gathered, returns)
    q_loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()
    return q_loss.detach(), gn

update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "returns"],
    out_keys=["td_loss", "gn"],
)

if __name__ == "__main__":
    ####### Argument Parsing #######
    args = tyro.cli(PQNArgs)
    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    
    args.log_iterations = np.linspace(0, args.num_iterations, num=args.num_logs, endpoint=False, dtype=int)
    args.log_iterations = np.unique(args.log_iterations)
    args.log_iterations = np.insert(args.log_iterations, 0, 1)
    args.log_iterations_img = np.linspace(0, args.num_iterations, num=args.num_img_logs, endpoint=False, dtype=int)
    args.log_iterations_img = np.unique(args.log_iterations_img)
    args.log_iterations_img = np.insert(args.log_iterations_img, 0, 1)
    
    run_name = f"PQN_CNN:{args.cnn_type}_SIZE:{args.network_size}_ENV:{args.env_id}_SEED:{args.seed}"
    args.run_name = run_name
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Wandb Setup #######
    wandb.init(
        project=f"debug-pqn",
        name=run_name + f"_{int(time.time())}",
        config=vars(args),
        save_code=True,
    )

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
    policy = agent_inference.forward
    get_max_value = agent_inference.get_max_value

    if args.compile:
        mode = None
        policy = torch.compile(policy, mode=mode)
        lambda_returns = torch.compile(lambda_returns, fullgraph=True, mode=mode)
        update = torch.compile(update, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        update = CudaGraphModule(update)

    ####### Training Loop #######
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

        # anneal learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        # collect rollout
        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(next_obs, next_done, avg_returns=avg_returns)
        global_step += container.numel()
        
        # compute lambda returns
        container = lambda_returns(next_obs, next_done, container)
        container_flat = container.view(-1)

        # update
        gns = []
        old_agent.load_state_dict(agent.state_dict())
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b_idx, b in enumerate(b_inds):
                container_local = container_flat[b]
                out = update(container_local, tensordict_out=tensordict.TensorDict())
                gns.append(out["gn"].cpu().numpy())

                # log images
                if epoch == 0 and b_idx == 0 and global_step_burnin is not None and iteration in args.log_iterations_img:
                    try:
                        compute_visitation_distribution(
                            agent,
                            container["obs"],
                            D=512,
                            nsteps=args.num_steps,
                            nenvs=args.num_envs,
                            env_id=args.env_id,
                            ep_return=np.array(avg_returns).mean()
                        )
                    except:
                        print("Failed to compute visitation distribution")
                        pass
                
                # log churn stats
                if epoch == 0 and b_idx == 0 and global_step_burnin is not None and iteration in args.log_iterations:
                    with torch.no_grad():
                        max_to_keep = min(4096, len(container_flat))
                        cntner_churn = container_flat[torch.randperm(len(container_flat))[:max_to_keep]]
                        churn_stats = compute_representation_and_q_churn(agent, old_agent, cntner_churn["obs"])
                        try:
                            ranks = compute_ranks_from_features(agent, cntner_churn["obs"])
                        except:
                            ranks = {}
               
        if global_step_burnin is not None and iteration in args.log_iterations:
            cur_time = time.time()
            speed = (global_step - global_step_burnin) / (cur_time - start_time)
            global_step_burnin = global_step
            start_time = cur_time
            avg_returns_t = torch.tensor(avg_returns).mean()
            pbar.set_description(
                f"global.step: {global_step: 8d}, "
                f"sps: {speed: 4.1f} sps, "
                f"avg.ep.return: {avg_returns_t: 4.2f},"
            )
        
            with torch.no_grad():
                ep_return = np.array(avg_returns).mean()
                logs = {
                    "charts/sps": speed,
                    "charts/episode_return": ep_return,
                    "charts/hns": _compute_hns(args.env_id, ep_return),
                    "losses/lambda_returns": container["returns"].mean(),
                    "losses/q_values": container["vals"].mean(),
                    "losses/td_loss": out["td_loss"],
                    "losses/gradient_norm": np.mean(gns),
                    "schedule/epsilon": linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step),
                    "schedule/lr": optimizer.param_groups[0]["lr"],
                    
                }
                logs = {**logs, **churn_stats, **ranks}

            wandb.log(
                {"charts/global_step": global_step, **logs}, step=global_step
            )

    envs.close()