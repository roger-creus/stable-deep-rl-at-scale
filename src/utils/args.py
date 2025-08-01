import os
from dataclasses import dataclass
    
@dataclass
class PQNArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    num_logs: int = 50
    """Number of logs to save."""
    num_img_logs: int = 10
    """Number of image logs to save."""
    wandb_project_id: str = "atari10-pqn"
    """the wandb project name"""

    # Learning rate schedule
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.005
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""
    
    # Agent Args
    use_ln: bool = True  # True, False
    """whether to use layer normalization"""
    use_spectral_norm: bool = False
    """whether to use spectral normalization"""
    cnn_type: str = "atari"
    """the type of the CNN"""
    mlp_type: str = "default" #default,residual,multiskip,densenet
    """the type of the MLP"""
    cnn_size: str = "medium" # small, medium, large
    """the size of the network"""
    mlp_width: str = "small" # small, medium, large
    """the size of the network"""
    mlp_depth: str = "small" # small, medium, large
    """the size of the network"""
    activation_fn: str = "relu"
    """the activation function of the network"""
    optimizer: str = "radam" # radam, kron
    """the optimizer of the network"""
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 1
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""
    
    
@dataclass
class PPOArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    num_logs: int = 50
    """Number of logs to save."""
    num_img_logs: int = 10
    """Number of image logs to save."""
    wandb_project_id: str = "atari10-ppo"
    """the wandb project name"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
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

    # Agent Args
    use_ln: bool = False
    """whether to use layer normalization"""
    cnn_type: str = "atari"  # atari, impala
    """the type of the CNN"""
    mlp_type: str = "default" # fully connected,residual,multiskip,densenet
    """the type of the MLP"""
    cnn_size: str = "medium" # small, medium, large
    """the size of the network"""
    mlp_width: str = "small" # small, medium, large
    """the size of the network"""
    mlp_depth: str = "small" # small, medium, large
    """the size of the network"""
    activation_fn: str = "relu"
    """the activation function of the network"""
    optimizer: str = "radam" # radam, kron
    """the optimizer of the network"""
    shared_trunk: bool = True
    """whether to use a shared trunk agent"""
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    measure_burnin: int = 1
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""
    