import torch
import torch.nn.functional as F
from IPython import embed

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
        "learning/feature_norm": feature_norm
    }
    
@torch.no_grad()
def compute_ppo_metrics(agent, obs):
    representation, dead_neurons = agent.get_representation(obs)
    
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
    