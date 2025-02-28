import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["WANDB__SERVICE_WAIT"] = "300"

import os
import random
import time
import envpool
import math
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
from collections import deque
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.categorical import Distribution

Distribution.set_default_validate_args(False)
torch.set_float32_matmul_precision("high")

from utils.compute_hns import _compute_hns
from utils.loss_landscape import plot_loss_landscape
from utils.utils import parse_cnn_size, parse_mlp_depth, parse_mlp_width, get_optimizer
from utils.args import PQNArgs
from utils.compute_churn import compute_representation_and_q_churn, compute_ranks_from_features, plot_representation_change, plot_activations_range
from utils.wrappers import RecordEpisodeStatistics
from models.agent import PQNAgent

from IPython import embed

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def get_alpha():
    return log_alpha.exp()

def lambda_returns(next_obs, next_done, container, alpha):
    next_value = get_softmax_value(next_obs, alpha)
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)
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
        i += 1
    returns = container["returns"] = torch.stack(list(reversed(returns)))
    return container

def rollout(obs, done, avg_returns=[], avg_lengths=[]):
    ts = []
    for step in range(args.num_steps):
        q_values = policy(obs)
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        explore = (torch.rand((args.num_envs,), device=device) < epsilon)
        random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,), device=device)
        
        alpha = get_alpha()
        policy_probs = F.softmax(q_values / alpha, dim=1)
        soft_actions = torch.multinomial(policy_probs, num_samples=1).squeeze(1)
        action = torch.where(explore, random_actions, soft_actions)
        
        next_obs_np, reward, next_done, info = envs.step(action.cpu().numpy())
        next_obs = torch.as_tensor(next_obs_np)
        reward = torch.as_tensor(reward)
        next_done = torch.as_tensor(next_done)
        
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
                vals=alpha * torch.logsumexp(q_values / alpha, dim=1),
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
    
    old_representation, per_layer_representations = agent.get_representation(obs, per_layer=True)
    old_val = agent.get_Q(old_representation)
    old_val_gathered = old_val.gather(1, actions.unsqueeze(1)).squeeze(1)
    q_loss = F.mse_loss(old_val_gathered, returns)
    q_loss.backward()
    
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
        elif "q" in k:
            new_clean_grad_norms[f"q"] = v
        else:
            new_clean_grad_norms[k] = v
    ################################################
    
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()
    return q_loss.detach(), gn, new_clean_grad_norms, per_layer_representations

update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "returns"],
    out_keys=["td_loss", "gn", "new_clean_grad_norms", "per_layer_representations"],
)

def update_alpha(obs, global_step):
    current_target_entropy = linear_schedule(
        target_entropy_init, target_entropy_end, args.target_entropy_fraction * args.total_timesteps, global_step
    )
    
    alpha = get_alpha()
    q_values = policy(obs)
    policy_probs = F.softmax(q_values / alpha, dim=1)
    log_policy_probs = F.log_softmax(q_values / alpha, dim=1)
    entropies = -(policy_probs * log_policy_probs).sum(dim=1)
    entropy = entropies.mean()
    
    alpha_loss = log_alpha * (entropy - current_target_entropy).detach()
    
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()
    return alpha_loss.detach(), entropy.detach(), current_target_entropy

update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "returns"],
    out_keys=["td_loss", "gn", "new_clean_grad_norms", "per_layer_representations"],
)

update_alpha = tensordict.nn.TensorDictModule(
    update_alpha,
    in_keys=["obs", "global_step"],
    out_keys=["alpha_loss", "entropy", "target_entropy"],
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
    
    run_name = f"SOFTPQN_ENV:{args.env_id}_OPTIM:{args.optimizer}_CNN:{args.cnn_type}_CNN.SIZE:{args.cnn_size}_MLP:{args.mlp_type}_MLP.WIDTH:{args.mlp_width}_MLP.DEPTH:{args.mlp_depth}_LN:{args.use_ln}_ACTFN:{args.activation_fn}_EPOCHS:{args.update_epochs}_SEED:{args.seed}"
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
    trunk_num_layers = parse_mlp_depth(args.mlp_depth, args.mlp_type)
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
    
    agent = PQNAgent(**agent_cfg)
    old_agent = PQNAgent(**agent_cfg)
    print(agent)
    
    agent_inference = PQNAgent(**agent_cfg)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)
    
    ###### Temperature fo Soft PQN ######
    initial_alpha = 1.0
    num_actions = envs.single_action_space.n
    target_entropy_init = -args.target_entropy_init_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
    target_entropy_end = -args.target_entropy_end_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
    print(f"Target Entropy Init: {target_entropy_init}, Target Entropy End: {target_entropy_end}")

    log_alpha = torch.tensor(math.log(initial_alpha), requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    ####### Optimizer #######
    # recommended in https://github.com/evanatyourservice/kron_torch
    if args.optimizer == "kron":
        args.learning_rate /= 3.0
        
    optimizer_clss = get_optimizer(args.optimizer)
    optimizer = optimizer_clss(
        agent.parameters(),
        lr=args.learning_rate,
    )

    ####### Executables #######
    policy = agent_inference.forward
    get_softmax_value = agent_inference.get_softmax_value
    
    if args.compile:
        mode = None
        policy = torch.compile(policy, mode=mode)
        lambda_returns = torch.compile(lambda_returns, fullgraph=True, mode=mode)
        update = torch.compile(update, mode=mode)
        update_alpha = torch.compile(update_alpha, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        update = CudaGraphModule(update)

    ####### Training Loop #######
    prev_container = None
    avg_returns = deque(maxlen=20)
    avg_lengths = deque(maxlen=20)
    
    layer_shapes_ = agent.get_layer_shapes()
    mu_representations = {
        k: torch.zeros(v, device=device) for k, v in layer_shapes_.items()
    }
    std_representations = {
        k: torch.ones(v, device=device) for k, v in layer_shapes_.items()
    }
    
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
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            next_obs, next_done, container = rollout(next_obs, next_done, avg_returns=avg_returns, avg_lengths=avg_lengths)
            global_step += container.numel()
            
            # compute lambda returns
            alpha = torch.as_tensor(get_alpha(), device=device)
            container = lambda_returns(next_obs, next_done, container, alpha)
            container_flat = container.view(-1)

        # update
        gns = []
        policy_entropies = []
        alpha_losses = []
        td_losses = []
        old_agent.load_state_dict(agent.state_dict())
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            
            for b_idx, b in enumerate(b_inds):
                container_local = container_flat[b]
                # update Q
                out = update(container_local, tensordict_out=tensordict.TensorDict())                   
                gns.append(out["gn"].cpu().numpy())
                td_losses.append(out["td_loss"].cpu().numpy())
                
                # update alpha
                alpha_loss, policy_entropy, current_target_entropy_ = update_alpha(container_local["obs"], torch.as_tensor(global_step, device=device))
                
                policy_entropies.append(policy_entropy.cpu().numpy())
                alpha_losses.append(alpha_loss.cpu().numpy())

                # log loss landscape
                if epoch == 0 and b_idx == 0 and global_step_burnin is not None and iteration in args.log_iterations_img and prev_container is not None:
                    with torch.no_grad():
                        max_to_keep = min(512, len(container_flat))
                        cntner_loss_landscape = container_flat[torch.randperm(len(container_flat))[:max_to_keep]]
                        batch_obs = cntner_loss_landscape['obs']
                        batch_actions = cntner_loss_landscape['actions']
                        batch_returns = cntner_loss_landscape['returns']
                        plot_loss_landscape(old_agent, batch_obs, batch_actions, batch_returns, grid_range=1.0, num_points=21, global_step=global_step)
                        
                # log churn stats
                if epoch == 0 and b_idx == 0 and global_step_burnin is not None and iteration in args.log_iterations:
                    with torch.no_grad():
                        max_to_keep = min(2048, len(container_flat))
                        cntner_churn = container_flat[torch.randperm(len(container_flat))[:max_to_keep]]
                        churn_stats = compute_representation_and_q_churn(agent, old_agent, cntner_churn["obs"])
                        per_layer_grad_norms = {
                            f"gradient_norms/{k}": v.item() for k, v in out["new_clean_grad_norms"].items()
                        }
                        
                        try:
                            ranks = compute_ranks_from_features(agent, cntner_churn["obs"])
                        except:
                            ranks = {}
                            
        # log representation change
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
                print(f"Failed to plot representation change: {e}")
        
        # logging   
        if global_step_burnin is not None and iteration in args.log_iterations:
            cur_time = time.time()
            speed = (global_step - global_step_burnin) / (cur_time - start_time)
            global_step_burnin = global_step
            start_time = cur_time
            avg_returns_t = torch.tensor(avg_returns).mean()
            avg_lengths_t = torch.tensor(avg_lengths).mean()
            pbar.set_description(
                f"global.step: {global_step: 8d}, "
                f"sps: {speed: 4.1f} sps, "
                f"avg.ep.return: {avg_returns_t: 4.2f},"
                f"avg.ep.length: {avg_lengths_t: 4.2f}"
            )
        
            with torch.no_grad():
                ep_return = np.array(avg_returns).mean() if len(avg_returns) > 0 else 0
                ep_length = np.array(avg_lengths).mean() if len(avg_lengths) > 0 else 0
                logs = {
                    "charts/sps": speed,
                    "charts/episode_return": ep_return,
                    "charts/episode_length": ep_length,
                    "charts/hns": _compute_hns(args.env_id, ep_return),
                    "alpha/alpha": get_alpha().item(),
                    "alpha/alpha_loss": np.mean(alpha_losses),
                    "alpha/policy_entropy": np.mean(policy_entropies),
                    "alpha/target_entropy": current_target_entropy_,
                    "losses/lambda_returns": container["returns"].mean(),
                    "losses/q_values": container["vals"].mean(),
                    "losses/td_loss": np.mean(td_losses),
                    "losses/gradient_norm": np.mean(gns),
                    "schedule/epsilon": linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step),
                    "schedule/lr": optimizer.param_groups[0]["lr"],
                    
                }
                logs = {**logs, **churn_stats, **ranks, **per_layer_grad_norms}

            wandb.log(
                {"charts/global_step": global_step, **logs}, step=global_step
            )

        # keep old batch for plots
        if iteration % 10 == 0:
            prev_container = container

    envs.close()