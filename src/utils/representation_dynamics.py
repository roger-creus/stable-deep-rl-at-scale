import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import umap
import numpy as np
from sklearn.decomposition import PCA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


"""
This script contains functions used for logging the learning dynamics of an agent in many different ways.
"""
    
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
    

# taken from https://github.com/CLAIRE-Labo/no-representation-no-trust/blob/dff26dd85df9fc7b2c5933ce67e35a0e1da21126/src/po_dynamics/modules/metrics.py
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

def compute_losslandscape_metrics(agent, obs, returns, actions, power_iters=20, trace_samples=50, eps=1e-3):
    """
    Returns:
      lambda_max : largest Hessian eigenvalue (scalar)
      trace_est  : Hutchinson’s estimator of Tr(H) (scalar)
    """
    predicted_qvalues = agent.get_Q(agent.get_representation(obs)).gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(predicted_qvalues, returns)
    
    params = [p for p in agent.network.parameters() if p.requires_grad]
    flat_params = torch.cat([p.view(-1) for p in params])
    num_params = flat_params.numel()
    
    def hvp(v):
        agent.network.zero_grad()
        grad_params = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_params])
        g_v = torch.dot(flat_grad, v)
        hv = torch.autograd.grad(g_v, params, retain_graph=True)
        return torch.cat([h.contiguous().view(-1) for h in hv])
    
    v = torch.randn(num_params, device=flat_params.device)
    v /= v.norm()
    for _ in range(power_iters):
        Hv = hvp(v)
        v = Hv.detach()
        v /= v.norm()
    Hv = hvp(v)
    lambda_max = torch.dot(v, Hv).item()
    
    trace_est = 0.0
    for _ in range(trace_samples):
        z = torch.randint(0, 2, (num_params,), device=flat_params.device, dtype=torch.float32)
        z[z == 0] = -1.0
        Hz = hvp(z)
        trace_est += torch.dot(z, Hz).item()
    trace_est /= trace_samples
    
    return {
        "loss_landscape/lambda_max": lambda_max,
        "loss_landscape/trace_est": trace_est,
    }