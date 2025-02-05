import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import umap
import numpy as np
from sklearn.decomposition import PCA
from IPython import embed

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@torch.no_grad()
def plot_representation_change(
    agent,
    old_agent,
    obs,
    prev_obs,
    global_step=0,
    num_points=200,
    name="learning_dynamics_change_per_iteration"
):
    # Unpack dimensions: (nsteps, nenvs, channels, height, width)
    nsteps, nenvs, C, H, W = obs.shape

    # Preallocate tensors for storing representations and value estimates.
    # For current obs batch:
    shapes = agent.get_layer_shapes()
    layer_names = list(shapes.keys())
    layer_shapes = list(shapes.values())
    layer_shapes = [torch.tensor(s).prod().item() for s in layer_shapes]
    
    agent_obs_representations = {
        layer_name: torch.zeros((nenvs, nsteps, layer_shape), device=obs.device) for layer_name, layer_shape in zip(layer_names, layer_shapes)
    }
    agent_obs_values = torch.zeros((nenvs, nsteps), device=obs.device)

    old_agent_obs_representations = {
        layer_name: torch.zeros((nenvs, nsteps, layer_shape), device=obs.device) for layer_name, layer_shape in zip(layer_names, layer_shapes)
    }
    old_agent_obs_values = torch.zeros((nenvs, nsteps), device=obs.device)
    
    # For previous obs:
    agent_obs_prev_representations = {
        layer_name: torch.zeros((nenvs, nsteps, layer_shape), device=obs.device) for layer_name, layer_shape in zip(layer_names, layer_shapes)
    }
    agent_obs_prev_values = torch.zeros((nenvs, nsteps), device=obs.device)
    
    old_agent_obs_prev_representations = {
        layer_name: torch.zeros((nenvs, nsteps, layer_shape), device=obs.device) for layer_name, layer_shape in zip(layer_names, layer_shapes)
    }
    old_agent_obs_prev_values = torch.zeros((nenvs, nsteps), device=obs.device)
    
    # Preallocate for optimal actions (for current observations)
    agent_opt_actions = torch.zeros((nenvs, nsteps), dtype=torch.long, device=obs.device)
    old_agent_opt_actions = torch.zeros((nenvs, nsteps), dtype=torch.long, device=obs.device)
    
    # Loop over each environment (or sequence) and compute representations, values, and optimal actions.
    for i in range(nenvs):
        # Get representations for current and previous observations.
        _, agent_obs_representation_ = agent.get_representation(obs[:, i], per_layer=True)
        for k, v in agent_obs_representation_.items():
            agent_obs_representations[k][i] = v.flatten(1)
        
        _, agent_obs_prev_representation_ = agent.get_representation(prev_obs[:, i], per_layer=True)
        for k, v in agent_obs_prev_representation_.items():
            agent_obs_prev_representations[k][i] = v.flatten(1)

        _, old_agent_obs_representation_ = old_agent.get_representation(obs[:, i], per_layer=True)
        for k, v in old_agent_obs_representation_.items():
            old_agent_obs_representations[k][i] = v.flatten(1)
        
        _, old_agent_obs_prev_representation_ = old_agent.get_representation(prev_obs[:, i], per_layer=True)
        for k, v in old_agent_obs_prev_representation_.items():
            old_agent_obs_prev_representations[k][i] = v.flatten(1)
        
        # Try computing the value estimates and optimal actions.
        try:
            pqn=False
            last_layer = list(agent_obs_representation_.keys())[-1]
            agent_obs_values[i] = agent.critic(agent_obs_representation_[last_layer]).squeeze(-1)
            agent_obs_prev_values[i] = agent.critic(agent_obs_prev_representation_[last_layer]).squeeze(-1)
            old_agent_obs_values[i] = old_agent.critic(old_agent_obs_representation_[last_layer]).squeeze(-1)
            old_agent_obs_prev_values[i] = old_agent.critic(old_agent_obs_prev_representation_[last_layer]).squeeze(-1)
        except Exception:
            pqn=True
            last_layer = list(agent_obs_representation_.keys())[-1]
            agent_obs_values[i] = agent.q_func(agent_obs_representation_[last_layer]).max(1).values.squeeze(-1)
            agent_obs_prev_values[i] = agent.q_func(agent_obs_prev_representation_[last_layer]).max(1).values.squeeze(-1)
            old_agent_obs_values[i] = old_agent.q_func(old_agent_obs_representation_[last_layer]).max(1).values.squeeze(-1)
            old_agent_obs_prev_values[i] = old_agent.q_func(old_agent_obs_prev_representation_[last_layer]).max(1).values.squeeze(-1)
            agent_q = agent.q_func(agent_obs_representation_[last_layer])
            old_agent_q = old_agent.q_func(old_agent_obs_representation_[last_layer])
        
        # Determine optimal actions (argmax over Q-values) for current observations.
        if pqn:
            agent_opt_actions[i] = agent_q.argmax(dim=1)
            old_agent_opt_actions[i] = old_agent_q.argmax(dim=1)
    
    # -------------------------------
    # Process data for CURRENT observations (obs)
    # -------------------------------
    agent_current_repr = {}
    for k,v in agent_obs_representations.items():
        agent_current_repr[k] = v.view(-1, v.shape[-1]).cpu().numpy()
    old_current_repr = {}
    for k,v in old_agent_obs_representations.items():
        old_current_repr[k] = v.view(-1, v.shape[-1]).cpu().numpy()
    
    agent_current_values = agent_obs_values.view(-1).cpu().numpy()
    old_current_values = old_agent_obs_values.view(-1).cpu().numpy()
    
    # Optimal actions for current obs.
    if pqn:
        agent_opt_actions_np = agent_opt_actions.view(-1).cpu().numpy()
        old_agent_opt_actions_np = old_agent_opt_actions.view(-1).cpu().numpy()
    
    # -- Quantitative Measures in the original embedding space --
    # L2 distances:
    l2_dists_current = {}
    for k in agent_current_repr.keys():
        l2_dists_current[k] = np.linalg.norm(agent_current_repr[k] - old_current_repr[k], axis=1)
    # Cosine similarities & distances:
    norm_agent = {}
    norm_old = {}
    for k in agent_current_repr.keys():
        norm_agent[k] = np.linalg.norm(agent_current_repr[k], axis=1)
        norm_old[k] = np.linalg.norm(old_current_repr[k], axis=1)
    cos_sim_current = {}
    cos_dists_current = {}
    for k in agent_current_repr.keys():
        cos_sim_current[k] = np.sum(agent_current_repr[k] * old_current_repr[k], axis=1) / (norm_agent[k] * norm_old[k] + 1e-8)
        cos_dists_current[k] = 1 - cos_sim_current[k]
        
    pca_results = {}
    marker_sizes = {}
    large_change_flags = {}
    action_changed_flags = {} if pqn else None
    delta_current_all = {}

    for k in agent_current_repr.keys():
        # Fit PCA on the concatenated current representations.
        pca = PCA(n_components=2)
        combined_repr = np.concatenate([agent_current_repr[k], old_current_repr[k]], axis=0)
        pca.fit(combined_repr)
        
        # Transform the representations.
        agent_pca = pca.transform(agent_current_repr[k])
        old_pca = pca.transform(old_current_repr[k])
        
        # Compute distances between current representations (in PCA space).
        delta_current = np.linalg.norm(agent_pca - old_pca, axis=1)
        delta_current_all[k] = delta_current
        
        # Scale marker sizes by change magnitude.
        marker_sizes[k] = 60 + 200 * (delta_current / (delta_current.max() + 1e-8))
        
        # Flag the largest 10% of changes.
        large_change_flags[k] = delta_current >= np.percentile(delta_current, 90)
        
        # Flag where the optimal action has changed.
        if pqn:
            action_changed_flags[k] = agent_opt_actions_np != old_agent_opt_actions_np
        
        # Store PCA results.
        pca_results[k] = (agent_pca, old_pca)

    # Subsample if needed.
    n_total = next(iter(agent_current_repr.values())).shape[0]
    idx = np.random.choice(n_total, num_points, replace=False)
    if n_total > num_points:
        for k in agent_current_repr.keys():
            agent_pca, old_pca = pca_results[k]
            pca_results[k] = (agent_pca[idx], old_pca[idx])
            marker_sizes[k] = marker_sizes[k][idx]
            large_change_flags[k] = large_change_flags[k][idx]
            delta_current_all[k] = delta_current_all[k][idx]
            if pqn:
                action_changed_flags[k] = action_changed_flags[k][idx]
        
        agent_current_values = agent_current_values[idx]
        old_current_values = old_current_values[idx]
        
        if pqn:
            agent_opt_actions_np = agent_opt_actions_np[idx]
            old_agent_opt_actions_np = old_agent_opt_actions_np[idx]
        
    # -------------------------------
    # Process data for PREVIOUS observations (prev_obs)
    # -------------------------------
    agent_prev_repr = {}
    for k,v in agent_obs_prev_representations.items():
        agent_prev_repr[k] = v.view(-1, v.shape[-1]).cpu().numpy()
    old_prev_repr = {}
    for k,v in old_agent_obs_prev_representations.items():
        old_prev_repr[k] = v.view(-1, v.shape[-1]).cpu().numpy()
    
    agent_prev_values = agent_obs_prev_values.view(-1).cpu().numpy()
    old_prev_values = old_agent_obs_prev_values.view(-1).cpu().numpy()
    
    # For optimal actions on previous observations, we need to compute them.
    if pqn:
        last_layer = list(agent_obs_prev_representation_.keys())[-1]
        last_shape = agent_obs_prev_representation_[last_layer].shape[-1]
        agent_prev_q = agent.q_func(agent_obs_prev_representations[last_layer].view(-1, last_shape))
        old_agent_prev_q = old_agent.q_func(old_agent_obs_prev_representations[last_layer].view(-1, last_shape))
        agent_opt_actions_prev = agent_prev_q.argmax(dim=1).cpu().numpy()
        old_agent_opt_actions_prev = old_agent_prev_q.argmax(dim=1).cpu().numpy()
    else:
        last_layer = list(agent_obs_prev_representation_.keys())[-1]
        last_shape = agent_obs_prev_representation_[last_layer].shape[-1]
        agent_prev_q = agent.critic(agent_obs_prev_representations[last_layer].view(-1, last_shape))
        old_agent_prev_q = old_agent.critic(old_agent_obs_prev_representations[last_layer].view(-1, last_shape))
    
    l2_dists_prev = {}
    for k in agent_prev_repr.keys():
        l2_dists_prev[k] = np.linalg.norm(agent_prev_repr[k] - old_prev_repr[k], axis=1)
    norm_agent_prev = {}
    norm_old_prev = {}
    for k in agent_prev_repr.keys():
        norm_agent_prev[k] = np.linalg.norm(agent_prev_repr[k], axis=1)
        norm_old_prev[k] = np.linalg.norm(old_prev_repr[k], axis=1)
    cos_sim_prev = {}
    cos_dists_prev = {}
    for k in agent_prev_repr.keys():
        cos_sim_prev[k] = np.sum(agent_prev_repr[k] * old_prev_repr[k], axis=1) / (norm_agent_prev[k] * norm_old_prev[k] + 1e-8)
        cos_dists_prev[k] = 1 - cos_sim_prev[k]

    pca_results_prev = {}
    marker_sizes_prev = {}
    large_change_flags_prev = {}
    action_changed_flags_prev = {} if pqn else None
    delta_prev_all = {}
    
    for k in agent_prev_repr.keys():
        pca = PCA(n_components=2)
        combined_repr = np.concatenate([agent_prev_repr[k], old_prev_repr[k]], axis=0)
        pca.fit(combined_repr)
        
        agent_prev_pca = pca.transform(agent_prev_repr[k])
        old_prev_pca = pca.transform(old_prev_repr[k])
        
        delta_prev = np.linalg.norm(agent_prev_pca - old_prev_pca, axis=1)
        delta_prev_all[k] = delta_prev
        
        marker_sizes_prev[k] = 60 + 200 * (delta_prev / (delta_prev.max() + 1e-8))
        large_change_flags_prev[k] = delta_prev >= np.percentile(delta_prev, 90)
        
        if pqn:
            action_changed_flags_prev[k] = agent_opt_actions_prev != old_agent_opt_actions_prev
        
        pca_results_prev[k] = (agent_prev_pca, old_prev_pca)
        
    if n_total > num_points:
        for k in agent_prev_repr.keys():
            agent_prev_pca, old_prev_pca = pca_results_prev[k]
            pca_results_prev[k] = (agent_prev_pca[idx], old_prev_pca[idx])
            marker_sizes_prev[k] = marker_sizes_prev[k][idx]
            large_change_flags_prev[k] = large_change_flags_prev[k][idx]
            delta_prev_all[k] = delta_prev_all[k][idx]
            
            if pqn:
                action_changed_flags_prev[k] = action_changed_flags_prev[k][idx]
        
        agent_prev_values = agent_prev_values[idx]
        old_prev_values = old_prev_values[idx]
        
        if pqn:
            agent_opt_actions_prev = agent_opt_actions_prev[idx]
            old_agent_opt_actions_prev = old_agent_opt_actions_prev[idx]
            
    
    # Loop over each representation key and create a plot.
    for k in agent_current_repr.keys():
        # Create a figure with 2 subplots (current on the left, previous on the right)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # -------------------------------
        # CURRENT Observations
        # -------------------------------
        # Unpack PCA results for current observations for key k.
        agent_current_pca, old_current_pca = pca_results[k]
        # Retrieve marker sizes and flags.
        marker_sizes_current = marker_sizes[k]
        large_change_current = large_change_flags[k]
        # For action change flag, if using pqn.
        if pqn:
            act_change_current = action_changed_flags[k]
        
        # Plot connecting lines for each point.
        for a_pt, o_pt in zip(agent_current_pca, old_current_pca):
            ax1.plot([o_pt[0], a_pt[0]], [o_pt[1], a_pt[1]], color='black', alpha=0.8, linewidth=1.25)
        
        # Plot the old agent points.
        sc1 = ax1.scatter(old_current_pca[:, 0], old_current_pca[:, 1],
                        marker='x', c=old_current_values, cmap='viridis',
                        label='Old Agent', s=marker_sizes_current)
        
        # Plot each current agent point individually to allow custom edge colors.
        for i in range(len(agent_current_pca)):
            if pqn:
                # Determine edge color.
                if large_change_current[i] and act_change_current[i]:
                    edge_color = 'purple'
                elif large_change_current[i]:
                    edge_color = 'red'
                elif act_change_current[i]:
                    edge_color = 'black'
                else:
                    edge_color = 'none'
            else:
                edge_color = 'none'
            
            # Normalize the agent current value for a color mapping.
            norm_val = (agent_current_values[i] - agent_current_values.min()) / (agent_current_values.ptp() + 1e-8)
            face_color = plt.cm.viridis(norm_val)
            ax1.scatter(agent_current_pca[i, 0], agent_current_pca[i, 1],
                        marker='o',
                        s=marker_sizes_current[i],
                        facecolor=face_color,
                        edgecolor=edge_color,
                        linewidth=2,
                        label='Agent' if i == 0 else "")
        
        ax1.set_title(f"Current Observations - {k}")
        ax1.set_xlabel("PCA Dimension 1")
        ax1.set_ylabel("PCA Dimension 2")
        ax1.legend()
        cbar1 = fig.colorbar(sc1, ax=ax1)
        cbar1.set_label("Old Agent Value Estimate")
        
        # -------------------------------
        # PREVIOUS Observations
        # -------------------------------
        # Unpack PCA results for previous observations for key k.
        agent_prev_pca, old_prev_pca = pca_results_prev[k]
        marker_sizes_prev_key = marker_sizes_prev[k]
        large_change_prev = large_change_flags_prev[k]
        if pqn:
            act_change_prev = action_changed_flags_prev[k]
        
        # Plot connecting lines.
        for a_pt, o_pt in zip(agent_prev_pca, old_prev_pca):
            ax2.plot([o_pt[0], a_pt[0]], [o_pt[1], a_pt[1]], color='black', alpha=0.8, linewidth=1.25)
        
        # Plot the old agent points.
        sc2 = ax2.scatter(old_prev_pca[:, 0], old_prev_pca[:, 1],
                        marker='x', c=old_prev_values, cmap='viridis',
                        label='Old Agent', s=marker_sizes_prev_key)
        
        # Plot each previous agent point individually with custom edge colors.
        for i in range(len(agent_prev_pca)):
            if pqn:
                if large_change_prev[i] and act_change_prev[i]:
                    edge_color = 'purple'
                elif large_change_prev[i]:
                    edge_color = 'red'
                elif act_change_prev[i]:
                    edge_color = 'black'
                else:
                    edge_color = 'none'
            else:
                edge_color = 'none'
            
            norm_val = (agent_prev_values[i] - agent_prev_values.min()) / (agent_prev_values.ptp() + 1e-8)
            face_color = plt.cm.viridis(norm_val)
            ax2.scatter(agent_prev_pca[i, 0], agent_prev_pca[i, 1],
                        marker='o',
                        s=marker_sizes_prev_key[i],
                        facecolor=face_color,
                        edgecolor=edge_color,
                        linewidth=2,
                        label='Agent' if i == 0 else "")
        
        ax2.set_title(f"Previous Observations - {k}")
        ax2.set_xlabel("PCA Dimension 1")
        ax2.set_ylabel("PCA Dimension 2")
        ax2.legend()
        cbar2 = fig.colorbar(sc2, ax=ax2)
        cbar2.set_label("Old Agent Value Estimate")
        plt.tight_layout()
        
        # Log the figure to wandb with a key that includes the current representation key.
        wandb.log({f"{k}_{name}/representation_change": wandb.Image(fig)}, step=global_step)
        wandb.log({
            f"{k}_{name}/mean_l2_distance_current": l2_dists_current[k].mean(),
            f"{k}_{name}/mean_cosine_distance_current": cos_dists_current[k].mean(),
            f"{k}_{name}/var_l2_distance_current": l2_dists_current[k].var(),
            f"{k}_{name}/var_cosine_distance_current": cos_dists_current[k].var(),
            f"{k}_{name}/mean_l2_distance_prev": l2_dists_prev[k].mean(),
            f"{k}_{name}/mean_cosine_distance_prev": cos_dists_prev[k].mean(),
            f"{k}_{name}/var_l2_distance_prev": l2_dists_prev[k].var(),
            f"{k}_{name}/var_cosine_distance_prev": cos_dists_prev[k].var(),
        }, step=global_step)
        plt.close(fig)
    
    wandb.log({
        f"{name}/policy_churn_current": 100 * np.mean(agent_opt_actions_np != old_agent_opt_actions_np) if pqn else 0,
        f"{name}/policy_churn_prev": 100 * np.mean(agent_opt_actions_prev != old_agent_opt_actions_prev) if pqn else 0,
    }, step=global_step)

@torch.no_grad()
def plot_visitation_distribution(
    agent,
    obs,
    D=512,
    nsteps=32,
    nenvs=128,
    env_id="Breakout-v5",
    global_step=0,
):
    nsteps, nenvs, C, H, W = obs.shape
    trajectories = torch.zeros((nenvs, nsteps, D), device=obs.device)
    values = torch.zeros((nenvs, nsteps), device=obs.device)
    for i in range(nenvs):
        representation = agent.get_representation(obs[:, i])
        if isinstance(representation, tuple):
            representation = representation[0]
        trajectories[i] = representation
        try:
            values[i] = agent.critic(representation).squeeze(-1)
        except:
            values[i] = agent.q_func(representation).max(1).values.squeeze(-1)
    
    reducer = umap.UMAP()
    trajectories_2d = reducer.fit_transform(trajectories.view(-1, D).cpu().numpy())
    values = values.cpu().numpy()
    fig, ax = plt.subplots(figsize=(15, 10))
    sc = ax.scatter(trajectories_2d[:, 0], trajectories_2d[:, 1], c=values.flatten(), cmap='viridis')
    fig.colorbar(sc, ax=ax)
    
    plt.title(f"{env_id} - Visitation Distribution - Step {global_step}")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.tight_layout()
    wandb.log({"learning_dynamics/visitation_distribution": wandb.Image(fig)}, step=global_step)
    plt.close(fig)

    
@torch.no_grad()
def compute_representation_and_q_churn(agent, old_agent, obs):
    # phi_t(x)
    old_hidden = old_agent.get_representation(obs)
    # psi_t(phi_t(x))
    old_Q = old_agent.get_Q(old_hidden)
    # phi_{t+1}(x)
    new_hidden, dead_neurons = agent.get_representation(obs, dead_neurons=True)
    # psi_{t+1}(phi_{t+1}(x))
    new_Q = agent.get_Q(new_hidden)

    # 1) policy churn ---> psi_t(phi_t(x)) != psi_{t+1}(phi_{t+1}(x))
    policy_churn = (new_Q.argmax(1) != old_Q.argmax(1)).float().mean()
    
    # 2) stability of Q-function -- psi_{t+1}(phi_{t+1}(x)) != psi_t(phi_{t+1}(x))
    old_Q_in_new_representation = old_agent.get_Q(new_hidden)
    q_value_stability = (new_Q.argmax(1) != old_Q_in_new_representation.argmax(1)).float().mean()
    
    # 3) stability of representations -- psi_{t+1}(phi_{t+1}(x)) != psi_{t+1}(phi_t(x))
    new_Q_in_old_representation = agent.get_Q(old_hidden)
    representation_stability = (new_Q.argmax(1) != new_Q_in_old_representation.argmax(1)).float().mean()
    
    # 4) change of policy due to change of representation -- psi_t(phi_t(x)) != psi_t(phi_{t+1}(x))
    change_of_policy_due_to_representation = (old_Q.argmax(1) != old_Q_in_new_representation.argmax(1)).float().mean()
    
    # 5) change of policy due to change of Q function -- psi_t(phi_t(x)) != psi_{t+1}(phi_t(x))
    change_of_policy_due_to_q_values = (old_Q.argmax(1) != new_Q_in_old_representation.argmax(1)).float().mean()
    
    # 6) cosine similarity between representations
    representations_cosine_similarity = F.cosine_similarity(old_hidden, new_hidden).mean()
    
    # 7) distance between representations (L2 distance)
    representations_l2_distance = F.pairwise_distance(old_hidden, new_hidden).mean()
    
    # 8) dead neurons in mlp
    dead_neurons_mlp = dead_neurons["mlp"]
    
    # 9) dead neurons in cnn
    dead_neurons_cnn = dead_neurons["cnn"]
    
    # 10) feature rank
    cov_matrix = torch.matmul(new_hidden.T, new_hidden) / new_hidden.shape[0]
    rank = torch.linalg.matrix_rank(cov_matrix)
    
    # 11) feature rank rankme
    eigvals = torch.linalg.eigvals(cov_matrix)
    normalized_eigvals = (eigvals / eigvals.sum()) + 1e-6
    entropy = -torch.sum(normalized_eigvals * torch.log(normalized_eigvals))
    rankme = torch.real(torch.exp(entropy)).cpu().item()
    
    # 12) feature norm
    feature_norm = torch.linalg.norm(new_hidden, ord=2, dim=1).mean()
    
    # 13) feature mean
    feature_mean = new_hidden.mean(dim=-1).mean(dim=0)
    
    # 14) feature std
    feature_std = new_hidden.std(dim=-1).mean(dim=0)
    
    return {
        "learning/policy_churn": policy_churn,
        "learning/representation_stability": representation_stability,
        "learning/q_value_stability": q_value_stability,
        "learning/change_of_policy_due_to_representation": change_of_policy_due_to_representation,
        "learning/change_of_policy_due_to_q_values": change_of_policy_due_to_q_values,
        "learning/representations_cosine_similarity": representations_cosine_similarity,
        "learning/representations_l2_distance": representations_l2_distance,
        "learning/dead_neurons_mlp": dead_neurons_mlp,
        "learning/dead_neurons_cnn": dead_neurons_cnn,
        "learning/feature_rank_torch": rank,
        "learning/feature_rank": rankme,
        "learning/feature_norm": feature_norm,
        "learning/feature_mean": feature_mean,
        "learning/feature_std": feature_std,
    }
    
@torch.no_grad()
def compute_ppo_metrics(agent, obs):
    representation, dead_neurons = agent.get_representation(obs, dead_neurons=True)
    
    # feature rank
    cov_matrix = torch.matmul(representation.T, representation) / representation.shape[0]
    rank = torch.linalg.matrix_rank(cov_matrix)
    
    # feature rank rankme
    eigvals = torch.linalg.eigvals(cov_matrix)
    normalized_eigvals = (eigvals / eigvals.sum()) + 1e-6
    entropy = -torch.sum(normalized_eigvals * torch.log(normalized_eigvals))
    rankme = torch.real(torch.exp(entropy)).cpu().item()
    
    # feature norm
    feature_norm = torch.linalg.norm(representation, ord=2, dim=1).mean()
    
    # dead neurons in mlp
    dead_neurons_mlp = dead_neurons["mlp"]
    dead_neurons_cnn = dead_neurons["cnn"]
    
    return {
        "learning/feature_rank_torch": rank,
        "learning/feature_rank": rankme,
        "learning/feature_norm": feature_norm,
        "learning/dead_neurons_mlp": dead_neurons_mlp,
        "learning/dead_neurons_cnn": dead_neurons_cnn,
    }
    
    
def compute_ranks_from_features(agent, obs):
    """Computes different approximations of the rank of the feature matrices.

    Args:
        feature_matrices (torch.Tensor): A tensor of shape (B_matrices, N_obs, D_dims).

    (1) Effective rank.
    A continuous approximation of the rank of a matrix.
    Definition 2.1. in Roy & Vetterli, (2007) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7098875
    Also used in Huh et al. (2023) https://arxiv.org/pdf/2103.10427.pdf


    (2) Approximate rank.
    Threshold at the dimensions explaining 99% of the variance in a PCA analysis.
    Section 2 in Yang et al. (2020) https://arxiv.org/pdf/1909.12255.pdf

    (3) srank.
    Another (incorrect?) version of (2).
    Section 3 in Kumar et al. https://arxiv.org/pdf/2010.14498.pdf

    (4) Feature rank.
    A threshold rank: normalize by dim size and discard dimensions with singular values below 0.01.
    Equations (4) and (5). Lyle et al. (2022) https://arxiv.org/pdf/2204.09560.pdf

    (5) PyTorch/NumPy rank.
    Rank defined in PyTorch and NumPy (https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html)
    (https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html)
    Quoting Numpy:
        This is the algorithm MATLAB uses [1].
        It also appears in Numerical recipes in the discussion of SVD solutions for linear least squares [2].
        [1] MATLAB reference documentation, “Rank” https://www.mathworks.com/help/techdoc/ref/rank.html
        [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, “Numerical Recipes (3rd edition)”,
        Cambridge University Press, 2007, page 795.

    """
    
    representation, dead_neurons = agent.get_representation(obs, dead_neurons=True)
    feature_matrices = representation.unsqueeze(0)
    
    cutoff = 0.01  # not used in (1), 1 - 99% in (2), delta in (3), epsilon in (4).
    threshold = 1 - cutoff

    svals = torch.linalg.svdvals(feature_matrices)

    # (1) Effective rank. Roy & Vetterli (2007)
    sval_sum = torch.sum(svals, dim=1)
    sval_dist = svals / sval_sum.unsqueeze(-1)
    # Replace 0 with 1. This is a safe trick to avoid log(0) = -inf
    # as Roy & Vetterli assume 0*log(0) = 0 = 1*log(1).
    sval_dist_fixed = torch.where(sval_dist == 0, torch.ones_like(sval_dist), sval_dist)
    effective_ranks = torch.exp(-torch.sum(sval_dist_fixed * torch.log(sval_dist_fixed), dim=1))

    # (2) Approximate rank. PCA variance. Yang et al. (2020)
    sval_squares = svals**2
    sval_squares_sum = torch.sum(sval_squares, dim=1)
    cumsum_squares = torch.cumsum(sval_squares, dim=1)
    threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum.unsqueeze(-1))
    approximate_ranks = (~threshold_crossed).sum(dim=-1) + 1

    # (3) srank. Weird. Kumar et al. (2020)
    cumsum = torch.cumsum(svals, dim=1)
    threshold_crossed = cumsum >= threshold * sval_sum.unsqueeze(-1)
    sranks = (~threshold_crossed).sum(dim=-1) + 1

    # (4) Feature rank. Most basic. Lyle et al. (2022)
    n_obs = torch.tensor(feature_matrices.shape[1], device=feature_matrices.device)
    svals_of_normalized = svals / torch.sqrt(n_obs)
    over_cutoff = svals_of_normalized > cutoff
    feature_ranks = over_cutoff.sum(dim=-1)

    # (5) PyTorch/NumPy rank.
    pytorch_ranks = torch.linalg.matrix_rank(feature_matrices)

    # Some singular values.
    singular_values = dict(
        lambda_1=svals_of_normalized[:, 0],
        lambda_N=svals_of_normalized[:, -1],
    )
    if svals_of_normalized.shape[1] > 1:
        singular_values.update(lambda_2=svals_of_normalized[:, 1])

    ranks = dict(
        effective_rank_vetterli=effective_ranks,
        approximate_rank_pca=approximate_ranks,
        srank_kumar=sranks,
        feature_rank_lyle=feature_ranks,
        pytorch_rank=pytorch_ranks,
    )

    out = {**singular_values, **ranks}
    out = {f"ranks/{k}": v for k, v in out.items()}
    return out