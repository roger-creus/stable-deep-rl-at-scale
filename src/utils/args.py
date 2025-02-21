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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    num_logs: int = 1000
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
    peak_lr: float = 3e-4
    """the peak learning rate for the learning rate scheduler"""
    end_lr: float = 1e-5
    """the end learning rate for the learning rate scheduler"""
    warmup_fraction: float = 0.1
    """the fraction of `total_timesteps` it takes from start_lr to peak_lr"""

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
    use_ln: bool = True
    """whether to use layer normalization"""
    cnn_type: str = "atari"
    """the type of the CNN"""
    mlp_type: str = "default" #default,residual
    """the type of the MLP"""
    cnn_size: str = "medium" # small, medium, large
    """the size of the network"""
    mlp_width: str = "small" # small, medium, large
    """the size of the network"""
    mlp_depth: str = "small" # small, medium, large
    """the size of the network"""
    activation_fn: str = "relu"
    """the activation function of the network"""
    optimizer: str = "adam" # adam, kron
    """the optimizer of the network"""
    
    # Soft PQN Args
    alpha_lr: float = 1e-4
    """the learning rate of the alpha parameter"""
    target_entropy_init_scale: float = 0.98
    """the maximum target entropy ratio"""
    target_entropy_end_scale: float = 0.05
    """the minimum target entropy ratio"""
    target_entropy_fraction: float = 0.8
    """the fraction of `total_timesteps` it takes from start to end"""
    
    # HLGauss PQN
    v_min: float = -10.0
    """the minimum value of the distribution"""
    v_max: float = 10.0
    """the maximum value of the distribution"""
    num_atoms: int = 51
    """the number of atoms in the distribution"""
    smoothing_ratio: float = 0.75
    """HLGauss sigma"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 1
    """Number of burn-in iterations for speed measure."""

    compile: bool = True
    """whether to use torch.compile."""
    cudagraphs: bool = True
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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    num_logs: int = 1000
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
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 3
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
    cnn_type: str = "atari"
    """the type of the CNN"""
    mlp_type: str = "default" #default,residual
    """the type of the MLP"""
    cnn_size: str = "medium" # small, medium, large
    """the size of the network"""
    mlp_width: str = "small" # small, medium, large
    """the size of the network"""
    mlp_depth: str = "small" # small, medium, large
    """the size of the network"""
    activation_fn: str = "relu"
    """the activation function of the network"""
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

    compile: bool = True
    """whether to use torch.compile."""
    cudagraphs: bool = True
    """whether to use cudagraphs on top of compile."""
    