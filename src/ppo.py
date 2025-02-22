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
from utils.utils import parse_cnn_size, parse_mlp_depth, parse_mlp_width, get_optimizer
from utils.args import PPOArgs
from utils.wrappers import RecordEpisodeStatistics
from utils.compute_churn import compute_ppo_metrics, compute_ranks_from_features, plot_representation_change
from utils.wrappers import RecordEpisodeStatistics
from models.agent import SharedTrunkPPOAgent, DecoupledPPOAgent

from IPython import embed

def gae(next_obs, next_done, container):
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

def rollout(obs, done, avg_returns=[], avg_lengths=[]):
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
                l = torch.as_tensor(info["l"]).float()
                avg_returns.extend(r[idx])
                avg_lengths.extend(l[idx])

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
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
    
    ######### log gradient norms per layer #########
    layer_grad_norms = {}
    for name, param in agent.named_parameters():
        if param.grad is not None and "weight" in name:
            if "cnn" in name:
                layer_grad_norms[name] = param.grad.detach().abs().mean()
            else:
                layer_grad_norms[name] = param.grad.detach().norm(2)

    clean_grad_norms = {k.replace("network.", "").replace(".weight", "").replace(".", "_"): v for k, v in layer_grad_norms.items()}
    clean_grad_norms_without_ln = {k: v for k, v in clean_grad_norms.items() if "ln" not in k}
    if len(clean_grad_norms_without_ln) == len(clean_grad_norms) and args.use_ln:
        clean_grad_norms = {k: v for i, (k, v) in enumerate(clean_grad_norms.items()) if i % 2 == 0}
    clean_grad_norms = {f"{k.split('_')[0]}_{i}": v for i, (k, v) in enumerate(clean_grad_norms.items())}
    c = 0
    new_clean_grad_norms = {}
    for k,v in clean_grad_norms.items():
        if "trunk" in k:
            new_clean_grad_norms[f"mlp_{c}"] = v
            c += 1
        elif "actor" in k:
            new_clean_grad_norms[f"actor"] = v
        elif "critic" in k:
            new_clean_grad_norms[f"critic"] = v
        else:
            new_clean_grad_norms[k] = v
    ################################################
    
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn, new_clean_grad_norms

update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn", "new_clean_grad_norms"],
)

if __name__ == "__main__":
    ####### Argument Parsing #######
    args = tyro.cli(PPOArgs)
    assert not (args.mlp_type == "residual" and args.mlp_size == "small"), "Residual MLP with small size is not supported"
    
    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    
    args.log_iterations = np.linspace(0, args.num_iterations, num=args.num_logs, endpoint=False, dtype=int)
    args.log_iterations = np.unique(args.log_iterations)
    args.log_iterations = np.insert(args.log_iterations, 0, 1)
    print(f"Will log {len(args.log_iterations)} times")
    
    args.log_iterations_img = np.linspace(0, args.num_iterations, num=args.num_img_logs, endpoint=False, dtype=int)
    args.log_iterations_img = np.unique(args.log_iterations_img)
    args.log_iterations_img = np.insert(args.log_iterations_img, 0, 1)
    print(f"Will log images {len(args.log_iterations_img)} times")
    
    run_name = f"PPO_ENV:{args.env_id}_OPTIM:{args.optimizer}_CNN:{args.cnn_type}_CNN.SIZE:{args.cnn_size}_MLP:{args.mlp_type}_MLP.WIDTH:{args.mlp_width}_MLP.DEPTH:{args.mlp_depth}_LN:{args.use_ln}_ACTFN:{args.activation_fn}_EPOCHS:{args.update_epochs}_SEED:{args.seed}"
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
    cnn_channels = parse_cnn_size(args.cnn_size)
    trunk_hidden_size = parse_mlp_width(args.mlp_width)
    trunk_num_layers = parse_mlp_depth(args.mlp_depth)
    agent_cfg = {
        "envs": envs,
        "use_ln": args.use_ln,
        "activation_fn": args.activation_fn,
        "cnn_type": args.cnn_type,
        "cnn_channels": cnn_channels,
        "mlp_type": args.mlp_type,
        "trunk_hidden_size": trunk_hidden_size,
        "trunk_num_layers": trunk_num_layers,
        "device": device,
    }
    agent_clss = SharedTrunkPPOAgent if args.shared_trunk else DecoupledPPOAgent
    
    agent = agent_clss(**agent_cfg)
    old_agent = agent_clss(**agent_cfg)
    print(agent)
    
    agent_inference = agent_clss(**agent_cfg)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    # recommended in https://github.com/evanatyourservice/kron_torch
    if args.optimizer == "kron":
        args.learning_rate /= 3.0
        
    optimizer_clss = get_optimizer(args.optimizer)
    optim_kwargs = {"eps": 1.0e-5} if "adam" in args.optimizer else {}
    
    optimizer = optimizer_clss(
        agent.parameters(),
        lr=args.learning_rate,
        **optim_kwargs,
    )

    ####### Executables #######
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile:
        mode = None
        policy = torch.compile(policy, mode=mode)
        gae = torch.compile(gae, fullgraph=True, mode=mode)
        update = torch.compile(update, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        update = CudaGraphModule(update)

    ####### Training Loop #######
    prev_container = None
    avg_returns = deque(maxlen=20)
    avg_lengths = deque(maxlen=20)
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
            optimizer.param_groups[0]["lr"] = lrnow

        # collect rollout
        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(next_obs, next_done, avg_returns=avg_returns, avg_lengths=avg_lengths)
        global_step += container.numel()

        # compute gae
        container = gae(next_obs, next_done, container)
        container_flat = container.view(-1)

        # update
        clipfracs = []
        kls= []
        gns = []
        old_agent.load_state_dict(agent.state_dict())
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b_idx, b in enumerate(b_inds):
                container_local = container_flat[b]
                out = update(container_local, tensordict_out=tensordict.TensorDict())
                
                clipfracs.append(out["clipfrac"].cpu().numpy())
                kls.append(out["approx_kl"].cpu().numpy())
                gns.append(out["gn"].cpu().numpy())
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
        
                # log churn stats
                if epoch == 0 and b_idx == 0 and global_step_burnin is not None and iteration in args.log_iterations:
                    with torch.no_grad():
                        max_to_keep = min(4096, len(container_flat))
                        cntner_churn = container_flat[torch.randperm(len(container_flat))[:max_to_keep]]
                        metrics = compute_ppo_metrics(agent, container_flat["obs"])
                        per_layer_grad_norms = {
                            f"gradient_norms/{k}": v.item() for k, v in out["new_clean_grad_norms"].items()
                        }
                        try:
                            ranks = compute_ranks_from_features(agent, container_flat["obs"])
                        except:
                            ranks = {}
        
        # log images                 
        # learning dynamics change per iteration
        if global_step_burnin is not None and iteration in args.log_iterations_img and prev_container is not None:
            try:
                plot_representation_change(
                    agent,
                    old_agent,
                    container["obs"],
                    prev_container["obs"],
                    global_step=global_step,
                    num_points=300,
                    name="learning_dynamics_change_per_iteration",
                )
            except Exception as e:
                print(f"Failed to compute learning dynamics plot per iteration: {e}")
                pass

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
                    "losses/entropy": out["entropy_loss"].mean(),
                    "losses/pg_loss": out["pg_loss"].mean(),
                    "losses/v_loss": out["v_loss"].mean(),
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
                logs = {**logs, **metrics, **ranks, **per_layer_grad_norms}

            wandb.log(
                {"charts/global_step": global_step, **logs}, step=global_step
            )
        
        # keep old batch for plots
        if iteration % 10 == 0:
            prev_container = container

    envs.close()
