# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["WANDB__SERVICE_WAIT"] = "300"

import os
import random
import time
from collections import deque
import envpool

import gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.categorical import Distribution

from utils.compute_hns import _compute_hns
from utils.utils import parse_network_size
from utils.args import PPOArgs
from utils.wrappers import RecordEpisodeStatistics
from utils.compute_churn import compute_ppo_metrics, compute_ranks_from_features, compute_visitation_distribution
from models.agent import SharedTrunkPPOAgent, DecoupledPPOAgent

Distribution.set_default_validate_args(False)
torch.set_float32_matmul_precision("high")

def gae(next_obs, next_done, container):
    # bootstrap value if not done
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done, avg_returns=[]):
    ts = []
    for step in range(args.num_steps):
        action, logprob, _, value = policy(obs=obs)
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
                # cleanrl ppo examples associate the done with the previous obs (not the done resulting from action)
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )

        obs = next_obs = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container


def update(obs, actions, logprobs, advantages, returns, vals):
    optimizer.zero_grad()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)

if __name__ == "__main__":
    args = tyro.cli(PPOArgs)
    batch_size = int(args.num_envs * args.num_steps)
    
    if args.minibatch_size == 0:
        batch_mode = "fixed_numMinibatches"
        args.minibatch_size = batch_size // args.num_minibatches
    else:
        batch_mode = "fixed_minibatchSize"
        args.num_minibatches = batch_size // args.minibatch_size

    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    args.log_iterations = np.linspace(0, args.num_iterations, num=args.num_logs, endpoint=False, dtype=int)
    args.log_iterations = np.unique(args.log_iterations)
    args.log_iterations = np.insert(args.log_iterations, 0, 1)
    args.log_iterations_img = np.linspace(0, args.num_iterations, num=args.num_img_logs, endpoint=False, dtype=int)
    args.log_iterations_img = np.unique(args.log_iterations_img)
    args.log_iterations_img = np.insert(args.log_iterations_img, 0, 1)
    
    
    run_name = f"PPO_CNN:{args.cnn_type}_SIZE:{args.network_size}_NENVS:{args.num_envs}_NSTEPS:{args.num_steps}_MODE:{batch_mode}_NEPOCHS:{args.update_epochs}_NMBS:{args.num_minibatches}_GAE:{args.gae_lambda}"
    args.run_name = run_name

    wandb.init(
        project=f"ppo_{batch_mode}",
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
    agent_clss = SharedTrunkPPOAgent if args.shared_trunk else DecoupledPPOAgent
    
    agent = agent_clss(**agent_cfg)
    print(agent)
    # Make a version of agent with detached params
    agent_inference = agent_clss(**agent_cfg)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        policy = torch.compile(policy, mode=mode)
        gae = torch.compile(gae, fullgraph=True, mode=mode)
        update = torch.compile(update, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        #gae = CudaGraphModule(gae)
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

        container = gae(next_obs, next_done, container)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        kls= []
        gns = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]
                out = update(container_local, tensordict_out=tensordict.TensorDict())
                
                clipfracs.append(out["clipfrac"].cpu().numpy())
                kls.append(out["approx_kl"].cpu().numpy())
                gns.append(out["gn"].cpu().numpy())
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break
        
        if global_step_burnin is not None and iteration in args.log_iterations_img:
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

        if global_step_burnin is not None and iteration in args.log_iterations:
            with torch.no_grad():
                max_len = min(4096, container_flat["obs"].shape[0])
                container_flat = container_flat[:max_len]
                metrics = compute_ppo_metrics(agent, container_flat["obs"])
                try:
                    ranks = compute_ranks_from_features(agent, container_flat["obs"])
                except:
                    ranks = {}

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
                    "entropy": out["entropy_loss"].mean(),
                    "pg_loss": out["pg_loss"].mean(),
                    "v_loss": out["v_loss"].mean(),
                    "approx_kl": np.mean(kls),
                    "clipfrac": np.mean(clipfracs),
                    "hns": _compute_hns(args.env_id, ep_return),
                    "logprobs": container["logprobs"].mean(),
                    "advantages": container["advantages"].mean(),
                    "returns": container["returns"].mean(),
                    "vals": container["vals"].mean(),
                    "gn": np.mean(gns),
                }
                logs = {**logs, **metrics, **ranks}

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
