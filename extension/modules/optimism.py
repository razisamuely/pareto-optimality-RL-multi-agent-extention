
import numpy as np
import torch

class OptimismScheduler:
    """
    Decays an optimism coefficient alpha from start_val to end_val over t_max steps.
    """
    def __init__(self, start_val=1.0, end_val=0.0, steps=100000, decay_type='linear'):
        self.start_val = start_val
        self.end_val = end_val
        self.total_steps = steps
        self.decay_type = decay_type.lower()
        self.current_step = 0

    def step(self):
        """Update the internal timer."""
        self.current_step += 1

    def get_alpha(self):
        """Calculate current alpha based on schedule."""
        if self.current_step >= self.total_steps:
            return self.end_val

        if self.decay_type == "linear":
            # Linear interpolation
            frac = self.current_step / float(self.total_steps)
            return self.start_val - frac * (self.start_val - self.end_val)
        
        elif self.decay_type == "exp":
            # Exponential decay: N(t) = N0 * e^(-lambda * t)
            # We want N(steps) ~ end_val. (If end_val=0, we target 0.01 or similar)
            # Simplification: alpha = start * (decay_rate ^ step)
            # Let's use a standard exp decay where at t=steps we are at 1% of range if end=0
            decay_rate = 0.99995 # Slow decay default, should probably be tuned or computed
            # Better: use parameter-free exponential-ish decay or strict mathematical definition
            # alpha = start * exp(-5 * t / T) -> at t=T, alpha ~ start * 0.006
            return self.start_val * np.exp(-5.0 * self.current_step / self.total_steps) + self.end_val
        
        else:
            return self.start_val

def cvar_q(q_values, alpha):
    """
    Computes the Conditional Value at Risk (CVaR) of the Q-value distribution.
    
    Args:
        q_values (torch.Tensor): Shape [batch, n_actions] or similar.
                                 Represents Q(s, a_i, a_{-i}) where the last dim is over a_{-i}.
        alpha (float): Quantile to average over (0 < alpha <= 1).
                       - alpha -> 0: Closer to Max
                       - alpha = 1: Mean
    
    Returns:
        torch.Tensor: Shape [batch] with the CVaR value.
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError("Alpha must be in (0, 1]")
    
    # Sort Q-values along the last dimension (actions of other agents)
    # q_values shape: [..., n_joint_actions]
    sorted_q, _ = torch.sort(q_values, descending=True, dim=-1)
    
    n_actions = q_values.shape[-1]
    k = int(np.ceil(alpha * n_actions))
    k = max(1, k) # Ensure at least 1
    
    # Take top k values
    top_k = sorted_q[..., :k]
    
    # Mean of top k
    return top_k.mean(dim=-1)
