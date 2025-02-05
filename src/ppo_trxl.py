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


Distribution.set_default_validate_args(False)
torch.set_float32_matmul_precision("high")

from utils.compute_hns import _compute_hns
from utils.utils import parse_network_size, find_all_modules, get_act_fn_clss
from utils.args import PPOArgs
from utils.wrappers import RecordEpisodeStatistics
from utils.compute_churn import compute_ppo_metrics, compute_ranks_from_features, plot_representation_change
from utils.wrappers import RecordEpisodeStatistics
from models.agent import TransformerPPOAgent
from rl_act import plot_rlact, log_rlact_parameters
from IPython import embed

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def gae(next_obs, next_done, container, next_memory, memory_mask, stored_memory_indices, env_current_episode_step, device):
    start = torch.clip(env_current_episode_step - args.trxl_memory_length, 0)
    end = torch.clip(env_current_episode_step, args.trxl_memory_length)
    indices = torch.stack([torch.arange(start[b], end[b], device=device) for b in range(args.num_envs)]).long()
    memory_window = batched_index_select(next_memory, 1, indices)

    next_value = agent.get_value(
        next_obs,
        memory_window,
        memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)],
        stored_memory_indices[-1],
    ).squeeze()
    
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

    advantages = torch.stack(list(reversed(advantages)))
    container["advantages"] = advantages
    container["returns"] = advantages + vals
    return container

def rollout(obs, done, avg_returns, avg_lengths, next_memory, memory_mask, memory_indices, env_current_episode_step):
    ts = []
    env_ids = torch.arange(args.num_envs)

    stored_memories = [next_memory[e] for e in range(args.num_envs)]
    stored_memory_index = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    for e in range(args.num_envs):
        stored_memory_index[:, e] = e
    
    for step in range(args.num_steps):
        current_memory_mask = memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)]
        current_memory_indices = memory_indices[env_current_episode_step]
        current_memory_window = batched_index_select(next_memory, 1, current_memory_indices)
        
        action, logprob, _, value, new_memory = policy(obs, current_memory_window, current_memory_mask, current_memory_indices)
        next_memory[env_ids, env_current_episode_step] = new_memory
        
        next_obs_np, reward, next_done, info = envs.step(action.cpu().numpy())

        next_obs = torch.as_tensor(next_obs_np, device=device)
        reward = torch.as_tensor(reward, device=device)
        next_done = torch.as_tensor(next_done, device=device)

        for id, d_ in enumerate(next_done):
            if d_:
                env_current_episode_step[id] = 0
                mem_index = stored_memory_index[step, id]
                stored_memories[mem_index] = stored_memories[mem_index].clone()
                next_memory[id] = torch.zeros(
                    (args.max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32
                )
                if step < args.num_steps - 1:
                    stored_memories.append(next_memory[id])
                    stored_memory_index[step + 1 :, id] = len(stored_memories) - 1
            else:
                env_current_episode_step[id] += 1

        idx = next_done
        if idx.any():
            idx = idx & torch.as_tensor(info["lives"] == 0, device=next_done.device, dtype=torch.bool)
            if idx.any():
                r = torch.as_tensor(info["r"], device=device)
                l = torch.as_tensor(info["l"], device=device).float()
                avg_returns.extend(r[idx].cpu().numpy())
                avg_lengths.extend(l[idx].cpu().numpy())
                
        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                dones=done,
                memory_mask=current_memory_mask,
                memory_indices=current_memory_indices,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,)
            )
        )

        obs = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container, next_memory, memory_mask, memory_indices, env_current_episode_step, stored_memories, stored_memory_index

def update(obs, actions, logprobs, advantages, returns, vals, mb_memory_windows, mb_memory_mask, mb_memory_indices):    
    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
        obs, mb_memory_windows, mb_memory_mask, mb_memory_indices, action=actions
    )
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss.
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss.
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(newvalue - vals, -args.clip_coef, args.clip_coef)
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    
    optimizer.zero_grad()
    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return (approx_kl, v_loss.detach(), pg_loss.detach(),
            entropy_loss.detach(), old_approx_kl, clipfrac, gn)

update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals", "mb_memory_windows", "mb_memory_mask", "mb_memory_indices"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)

if __name__ == "__main__":
    ####### Argument Parsing #######
    args = tyro.cli(PPOArgs)
    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size

    # Setup logging iterations (for printing/logging)
    args.log_iterations = np.linspace(0, args.num_iterations, num=args.num_logs, endpoint=False, dtype=int)
    args.log_iterations = np.unique(args.log_iterations)
    args.log_iterations = np.insert(args.log_iterations, 0, 1)
    args.log_iterations_img = np.linspace(0, args.num_iterations, num=args.num_img_logs, endpoint=False, dtype=int)
    args.log_iterations_img = np.unique(args.log_iterations_img)
    args.log_iterations_img = np.insert(args.log_iterations_img, 0, 1)
    
    # transformer args
    args.trxl_memory_length = min(args.trxl_memory_length, args.max_episode_steps)

    run_name = f"PPOTRXL_ENV:{args.env_id}_CNN:{args.cnn_type}_SIZE:{args.network_size}_LN:{args.use_ln}_ACTFN:{args.activation_fn}_SEED:{args.seed}"
    args.run_name = run_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Wandb Setup #######
    wandb.init(
        project=args.wandb_project_id,
        name=run_name + f"_{int(time.time())}",
        config=vars(args),
        save_code=True,
    )

    ####### Environment Setup #######
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    ####### Agent Setup #######
    cnn_channels, trunk_hidden_size, trunk_num_layers = parse_network_size(args.network_size)
    agent_cfg = {
        "envs": envs,
        "use_ln": args.use_ln,
        "activation_fn": args.activation_fn,
        "cnn_type": args.cnn_type,
        "cnn_channels": cnn_channels,
        "device": device,
        "trxl_dim": args.trxl_dim,
        "trxl_num_layers": args.trxl_num_layers,
        "trxl_num_heads": args.trxl_num_heads,
        "trxl_positional_encoding": args.trxl_positional_encoding,
    }
    # Choose your PPO-TRXL agent class (for example, PPOTrXLAgent)
    agent = TransformerPPOAgent(**agent_cfg)
    old_agent = TransformerPPOAgent(**agent_cfg)
    print(agent)

    # Inference agent setup.
    agent_inference = TransformerPPOAgent(**agent_cfg)
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
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    if args.compile:
        mode = None
        policy = torch.compile(policy, mode=mode)
        gae_compiled = torch.compile(gae, fullgraph=True, mode=mode)
        #update = torch.compile(update, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        #update = CudaGraphModule(update)

    ####### Training Loop #######
    avg_returns = deque(maxlen=20)
    avg_lengths = deque(maxlen=20)
    global_step = 0
    container_local = None
    next_obs = torch.tensor(envs.reset(), device=device, dtype=torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)

    next_memory = torch.zeros((args.num_envs, args.max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32, device=device)
    memory_mask = torch.tril(torch.ones((args.trxl_memory_length, args.trxl_memory_length), device=device), diagonal=-1)
    repetitions = torch.repeat_interleave(torch.arange(0, args.trxl_memory_length, device=device).unsqueeze(0), args.trxl_memory_length - 1, dim=0).long()
    memory_indices = torch.stack([torch.arange(i, i + args.trxl_memory_length, device=device) for i in range(args.max_episode_steps - args.trxl_memory_length + 1)]).long()
    memory_indices = torch.cat((repetitions, memory_indices))

    env_ids = range(args.num_envs)
    env_current_episode_step = torch.zeros((args.num_envs,), dtype=torch.long)

    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Anneal learning rate if applicable.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        # no grad during rollot + gae
        with torch.no_grad():

            # rollout phase: collect trajectories.
            torch.compiler.cudagraph_mark_step_begin()
            next_obs, next_done, container, next_memory, memory_mask, memory_indices, env_current_episode_step, stored_memories, stored_memory_index = rollout(
                next_obs,
                next_done,
                avg_returns=avg_returns,
                avg_lengths=avg_lengths,
                next_memory=next_memory,
                memory_mask=memory_mask,
                memory_indices=memory_indices,
                env_current_episode_step=env_current_episode_step,
            )
            global_step += container.numel()

            # GAE phase: compute advantages and returns
            container = gae(next_obs, next_done, container, next_memory, memory_mask, container["memory_indices"], env_current_episode_step, device)
            container_flat = container.view(-1)
            stored_memories = torch.stack(stored_memories, dim=0)
            b_memory_index = stored_memory_index.reshape(-1).to(device)
        
        # remove unnecessary padding from TrXL memory, if applicable
        actual_max_episode_steps = int((container["memory_indices"] * container["memory_mask"]).max().item() + 1)
        if actual_max_episode_steps < args.trxl_memory_length:
            container_flat["memory_indices"] = container_flat["memory_indices"][:, :actual_max_episode_steps]
            container_flat["memory_mask"] = container_flat["memory_mask"][:, :actual_max_episode_steps]
            stored_memories = stored_memories[:, :actual_max_episode_steps]
        
        # Update phase: use mini-batches.
        clipfracs = []
        kls = []
        gns = []
        old_agent.load_state_dict(agent.state_dict())
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b_idx, b in enumerate(b_inds):
                container_local = container_flat[b]
                mb_memories = stored_memories[b_memory_index[b]]
                mb_memory_windows = batched_index_select(mb_memories, 1, container_local["memory_indices"])
                
                approx_kl, v_loss, pg_loss, entropy_loss, old_approx_kl, clipfrac, gn  = update(
                    container_local.get("obs"),
                    container_local.get("actions"),
                    container_local.get("logprobs"),
                    container_local.get("advantages"),
                    container_local.get("returns"),
                    container_local.get("vals"),
                    mb_memory_windows,
                    container_local.get("memory_mask"),
                    container_local.get("memory_indices")
                )
                    
                clipfracs.append(clipfrac.cpu().numpy())
                kls.append(approx_kl.cpu().numpy())
                gns.append(gn.cpu().numpy())

        if global_step_burnin is not None and iteration in args.log_iterations:
            cur_time = time.time()
            speed = (global_step - global_step_burnin) / (cur_time - start_time)
            global_step_burnin = global_step
            start_time = cur_time

            r = container["rewards"].mean()
            r_max = container["rewards"].max()
            avg_returns_t = torch.tensor(avg_returns).mean()
            avg_lengths_t = torch.tensor(avg_lengths).mean()
            pbar.set_description(
                f"global.step: {global_step: 8d}, "
                f"sps: {speed: 4.1f} sps, "
                f"avg.ep.return: {avg_returns_t: 4.2f},"
                f"avg.ep.length: {avg_lengths_t: 4.2f},"
            )

            with torch.no_grad():
                ep_return = np.array(avg_returns).mean() if len(avg_returns) > 0 else 0
                ep_length = np.array(avg_lengths).mean() if len(avg_lengths) > 0 else 0
                logs = {
                    "charts/sps": speed,
                    "charts/episode_return": ep_return,
                    "charts/episode_length": ep_length,
                    "losses/entropy": entropy_loss,
                    "losses/pg_loss": pg_loss,
                    "losses/v_loss": v_loss,
                    "losses/approx_kl": np.mean(kls),
                    "losses/clipfrac": np.mean(clipfracs),
                    "charts/hns": _compute_hns(args.env_id, ep_return),
                    "losses/logprobs": container["logprobs"].mean(),
                    "losses/advantages": container["advantages"].mean(),
                    "losses/returns": container["returns"].mean(),
                    "losses/q_values": container["vals"].mean(),
                    "losses/gradient_norm": np.mean(gns),
                    "schedule/lr": optimizer.param_groups[0]["lr"],
                }

            wandb.log(
                {"charts/global_step": global_step, **logs}, step=global_step
            )
        
    envs.close()

