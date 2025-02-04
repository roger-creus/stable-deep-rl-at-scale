import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import wandb

class RLAct(nn.Module):
    def __init__(self, a=1.0, b=1.0, c=1.0, gamma=1.0, epsilon=1e-6):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x):
        neg = self.a * (torch.exp(x / self.b) - 1)
        mid = x
        safe_x = torch.clamp(x - self.gamma, min=self.epsilon)
        pos = self.gamma + self.c * torch.log(1 + safe_x / self.c)
        out = torch.where(x < 0, neg, mid)
        out = torch.where(x > self.gamma, pos, out)
        return out
      
class Meta_ADARL(nn.Module):
    def __init__(self, init_a=1.0, init_b=1.0, init_c=1.0, init_gamma=1.0, epsilon=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(init_c, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.epsilon = epsilon  # To avoid log(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        self.a.data.clamp_(min=1e-4, max=10.0)
        self.b.data.clamp_(min=1e-4, max=10.0)
        self.c.data.clamp_(min=1e-4, max=10.0)
        self.gamma.data.clamp_(min=1e-4, max=10.0)
        
        neg = self.a * (torch.exp(x / self.b) - 1)
        mid = x
        safe_x = torch.clamp(x - self.gamma, min=self.epsilon)
        pos = self.gamma + self.c * torch.log(1 + safe_x / self.c)
        out = torch.where(x < 0, neg, mid)
        out = torch.where(x > self.gamma, pos, out)
        return out

    def to(self, device):
        super().to(device)
        self.a.data = self.a.data.to(device)
        self.b.data = self.b.data.to(device)
        self.c.data = self.c.data.to(device)
        self.gamma.data = self.gamma.data.to(device)
        return self
    
class Smooth_Meta_ADARL(nn.Module):
    def __init__(self, init_a=1.0, init_b=1.0, init_c=1.0, init_gamma=1.0, 
                 epsilon=1e-6, k=10.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(init_c, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.epsilon = epsilon  # to avoid log(0)
        self.k = k  # controls softness of gating
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x):
        # Clamp parameters to keep them within a reasonable range.
        self.a.data.clamp_(min=1e-4, max=10.0)
        self.b.data.clamp_(min=1e-4, max=10.0)
        self.c.data.clamp_(min=1e-4, max=10.0)
        self.gamma.data.clamp_(min=1e-4, max=10.0)
        
        # Negative branch: for very negative x
        f_neg = self.a * (torch.exp(x / self.b) - 1)
        # Linear branch: identity mapping
        f_lin = x
        # Positive branch: for x greater than gamma
        safe_x = torch.clamp(x - self.gamma, min=self.epsilon)
        f_pos = self.gamma + self.c * torch.log(1 + safe_x / self.c)
        
        # Define smooth gating functions using a sigmoid:
        w_neg = torch.sigmoid(-self.k * x)       # ~1 when x is very negative, ~0 when x>=0
        w_pos = torch.sigmoid(self.k * (x - self.gamma))  # ~0 when x << gamma, ~1 when x >> gamma
        
        # Blend the linear branch and positive branch smoothly:
        f_blend = (1 - w_pos) * f_lin + w_pos * f_pos
        
        # Finally, blend with the negative branch:
        output = w_neg * f_neg + (1 - w_neg) * f_blend
        return output

    def to(self, device):
        super().to(device)
        self.a.data = self.a.data.to(device)
        self.b.data = self.b.data.to(device)
        self.c.data = self.c.data.to(device)
        self.gamma.data = self.gamma.data.to(device)
        return self

class Heuristic_ADARL(nn.Module):
    def __init__(self, tau=0.01, init_a=0.1, init_b=1.0, init_c=1.0, init_gamma=1.0, epsilon=1e-6):
        super().__init__()
        
        # Store as plain tensors (ensure they’re moved to correct device externally)
        self.a = torch.tensor(init_a, dtype=torch.float32)
        self.b = torch.tensor(init_b, dtype=torch.float32)
        self.c = torch.tensor(init_c, dtype=torch.float32)
        self.gamma = torch.tensor(init_gamma, dtype=torch.float32)
        self.epsilon = epsilon

        # Running estimates (initialize gamma running estimate as gamma)
        self.running_mean = 0.0
        self.running_std = 1.0
        self.running_gamma = self.gamma.clone()
        self.tau = tau

    def forward(self, x):
        # Activation function
        neg = self.a * (torch.exp(x / self.b) - 1)
        mid = x
        safe_x = torch.clamp(x - self.gamma, min=self.epsilon)
        pos = self.gamma + self.c * torch.log(1 + safe_x / self.c)
        # Use piecewise selection
        out = torch.where(x < 0, neg, mid)
        out = torch.where(x > self.gamma, pos, out)
        return out

    @torch.no_grad()
    def adapt(self, td_errors):
        # Compute instantaneous statistics
        mean_td = td_errors.mean()
        std_td = td_errors.std()

        # Fallback: if no negative or positive values, use overall stats
        neg_mask = td_errors < 0
        pos_mask = td_errors > 0
        if neg_mask.sum() >= 2:
            neg_std = td_errors[neg_mask].std()
            neg_mean = td_errors[neg_mask].mean()
        else:
            neg_std = std_td
            neg_mean = mean_td

        if pos_mask.sum() >= 2:
            pos_std = td_errors[pos_mask].std()
        else:
            pos_std = std_td

        # Update running estimates for overall TD errors (for potential future use)
        self.running_mean = (1 - self.tau) * self.running_mean + self.tau * mean_td
        self.running_std = (1 - self.tau) * self.running_std + self.tau * std_td

        # Update gamma using a running quantile (90th percentile)
        quantile_val = torch.quantile(td_errors, 0.90)
        self.running_gamma = (1 - self.tau) * self.running_gamma + self.tau * quantile_val
        self.gamma = self.running_gamma.clone()

        # Update other parameters using a mix of instantaneous and running estimates
        self.a = (1 - self.tau) * self.a + self.tau * neg_std
        self.b = (1 - self.tau) * self.b + self.tau * neg_mean.abs()
        self.c = (1 - self.tau) * self.c + self.tau * pos_std

        # Optionally, you could clip the values of these parameters to prevent them
        # from growing too extreme:
        self.a.clamp_(min=1e-4, max=10.0)
        self.b.clamp_(min=1e-4, max=10.0)
        self.c.clamp_(min=1e-4, max=10.0)
        self.gamma.clamp_(min=1e-4, max=10.0)

                
def plot_rlact(all_act_fns, module_clss, ncols=2, global_step=0):
    x = torch.linspace(-10, 10, 1000).to("cuda" if torch.cuda.is_available() else "cpu")
    n_acts = len(all_act_fns)
    nrows = math.ceil(n_acts / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()  # flatten in case we have a grid
    
    for i, m in enumerate(all_act_fns):
        # Extract parameters, defaulting if not present
        a_val = m.a.item() if hasattr(m, "a") else 1.0
        b_val = m.b.item() if hasattr(m, "b") else 1.0
        c_val = m.c.item() if hasattr(m, "c") else 1.0
        gamma_val = m.gamma.item() if hasattr(m, "gamma") else 1.0

        # Create a temporary activation function instance with the extracted parameters
        act_fn = module_clss(a_val, b_val, c_val, gamma_val)
        y = act_fn(x)

        ax = axes[i]
        ax.plot(x.cpu().numpy(), y.cpu().detach().numpy())
        ax.set_title(f"{module_clss.__name__} {i}\n(a={a_val:.3f}, b={b_val:.3f}, c={c_val:.3f}, γ={gamma_val:.3f})")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
    
    # Remove any empty subplots if n_acts is not a multiple of ncols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    wandb.log({
      "activation/RL_ACT": wandb.Image(fig) 
    }, step=global_step)
    
def log_rlact_parameters(all_act_fns, global_step=0):
    for i, m in enumerate(all_act_fns):
        # Extract parameters, defaulting if not present
        a_val = m.a.item() if hasattr(m, "a") else 1.0
        b_val = m.b.item() if hasattr(m, "b") else 1.0
        c_val = m.c.item() if hasattr(m, "c") else 1.0
        gamma_val = m.gamma.item() if hasattr(m, "gamma") else 1.0
        wandb.log({
            f"activation/{m.__class__.__name__}_{i}": {
                "a": a_val,
                "b": b_val,
                "c": c_val,
                "gamma": gamma_val
            }
        }, step=global_step)