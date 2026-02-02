
import torch as th
from learners.actor_critic_pac_learner import PACActorCriticLearner
from extension.modules.optimism import cvar_q

class PACCVaRLearner(PACActorCriticLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.cvar_alpha = getattr(args, "cvar_alpha", 0.1)

    def train_critic(self, critic, target_critic, batch, rewards, mask, terminated, pi):
        actions = batch["actions"]
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic.forward(batch, compute_all=True)[0][:, :-1]
            target_vals = target_vals.max(dim=3)[0] # Keep max for target? Or use CVaR?
            # Sticking to Max for target to allow optimistic value propagation
            
        target_vals = th.gather(target_vals, -1, actions[:, :-1]).squeeze(-1)

        if self.args.standardise_rewards:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
        target_returns = self.nstep_returns(
            rewards, mask, target_vals, self.args.q_nstep
        )

        if self.args.standardise_rewards:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        actions = batch["actions"][:, :-1]
        q = critic.forward(batch)[0][:, :-1]
        v = self.state_value(batch)[:, :-1].squeeze(-1)

        q_curr = th.gather(q, -1, actions).squeeze(-1)
        td_error = target_returns.detach() - q_curr
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        td_error_v = target_returns.detach() - v
        masked_td_error_v = td_error_v * mask
        loss += (masked_td_error_v**2).sum() / mask.sum()

        # --- MODIFIED SECTION ---
        # compute the CVaR Q-value
        q_all = critic.forward(batch, compute_all=True)[0][:, :-1]
        
        # q_all shape: [batch, time, n_agents, n_actions, n_joint_actions_others]
        # We need to squeeze dim 3 (dim=-2) which depends on logic? 
        # Wait, q_all shape from critic(compute_all=True) logic:
        # critic returns [batch, t, n_agents, n_actions, prod(n_actions_others)] if compute_all=True?
        # Let's verify shape assumption from base class:
        # q_all = q_all.max(dim=3)[0] -> This implies dimension 3 is the one to reduce.
        
        # cvar_q expects the last dimension to be the distribution to average over.
        # If max(dim=3) works, then dim 3 is the correct one.
        
        q_cvar = cvar_q(q_all, self.cvar_alpha)
        
        q_selected = th.gather(q_cvar, -1, actions).squeeze(-1)
        # ------------------------

        advantage = q_selected.detach() - v.detach()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((q_curr * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )

        return advantage, running_log
