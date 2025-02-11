import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from IPython import embed

def plot_loss_landscape(agent, obs, actions, returns, grid_range=1.0, num_points=21, global_step=None):
    """
    Plots the loss landscape (a 3D surface) of the current agent parameters.
    
    Parameters:
        obs (torch.Tensor): Batch of observations.
        actions (torch.Tensor): Batch of actions.
        returns (torch.Tensor): Batch of returns (target Q-values).
        grid_range (float): The range of perturbation coefficients (default: 1.0).
        num_points (int): Number of grid points per axis (default: 21).
    """
    # 1. Save the current parameters as the base state.
    base_state = copy.deepcopy(agent.state_dict())
    
    # 2. Create two random directions.
    def get_random_direction():
        direction = {}
        for name, param in agent.named_parameters():
            if param.requires_grad:
                # Generate a random tensor with the same shape.
                direction[name] = torch.randn_like(param)
        return direction

    def normalize_direction(direction, state):
        normalized = {}
        for name, param in state.items():
            if name in direction:
                # Scale the random direction to match the norm of the parameter.
                norm_param = param.norm()
                norm_dir = direction[name].norm()
                if norm_dir > 0:
                    normalized[name] = direction[name] / norm_dir * norm_param
                else:
                    normalized[name] = direction[name]
        return normalized

    direction1 = normalize_direction(get_random_direction(), base_state)
    direction2 = normalize_direction(get_random_direction(), base_state)
    
    # 3. Define a function that perturbs the parameters and evaluates the loss.
    def evaluate_loss(a, b):
        # Create a new state by adding a * direction1 + b * direction2.
        new_state = {}
        for name, param in base_state.items():
            if name in direction1:
                new_state[name] = param + a * direction1[name] + b * direction2[name]
            else:
                new_state[name] = param
        # Load the new state into the agent.
        agent.load_state_dict(new_state)
        
        # Evaluate the loss on the batch.
        # (This code mirrors your update() loss computation.)
        with torch.no_grad():
            rep = agent.get_representation(obs)
            q_vals = agent.get_Q(rep)
            # Assuming actions is a 1D tensor of indices.
            q_vals_gathered = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q_vals_gathered, returns)
        return loss.item()

    # 4. Evaluate the loss on a grid in the (a,b) space.
    a_values = np.linspace(-grid_range, grid_range, num_points)
    b_values = np.linspace(-grid_range, grid_range, num_points)
    loss_values = np.zeros((num_points, num_points))
    
    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            loss_values[i, j] = evaluate_loss(a, b)
    
    # 5. Reset the agent's parameters back to the original state.
    agent.load_state_dict(base_state)
    
    # 6. Plot the loss landscape in 3D.
    A, B = np.meshgrid(a_values, b_values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(A, B, loss_values.T, cmap='plasma', edgecolor='none')
    # null x and y labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title('Q Loss Landscape')
    plt.tight_layout()
    wandb.log({"loss_landscape/loss_landscape": wandb.Image(fig)}, step=global_step)
    plt.close(fig)