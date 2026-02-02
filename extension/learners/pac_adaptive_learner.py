
import torch as th
from learners.actor_critic_pac_learner import PACActorCriticLearner
from extension.modules.optimism import OptimismScheduler

class PACAdaptiveLearner(PACActorCriticLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        
        # Initialize the scheduler
        self.scheduler = OptimismScheduler(
            start_val=getattr(args, "optimism_start", 1.0),
            end_val=getattr(args, "optimism_end", 0.0),
            steps=getattr(args, "optimism_decay_steps", 50000),
            decay_type=getattr(args, "optimism_schedule", "linear")
        )

    def train(self, batch, t_env, episode_num):
        # Update scheduler
        self.scheduler.step()
        
        # Log alpha
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("optimism_alpha", self.scheduler.get_alpha(), t_env)
            
        super().train(batch, t_env, episode_num)

    def train_critic(self, critic, target_critic, batch, rewards, mask, terminated, pi):
        # We need to mostly copy this because the original class doesn't expose a 'get_optimistic_q' method.
        # Ideally, we would refactor the base class, but we are restricted to extensions.
        
        actions = batch["actions"]
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic.forward(batch, compute_all=True)[0][:, :-1]
            target_vals = target_vals.max(dim=3)[0] # Target is still max Q? Or should it be Adaptive too?
            # Standard PAC usually keeps target as Max Q (Optimistic Bellman), 
            # but for equilibrium selection, using the adaptive measure in target might also make sense.
            # However, the paper usually implies modifying the Actor's advantage estimation.
            # Let's keep target as standard Max Q for stability, affecting only the Advantage (Policy Gradient).

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
        # compute the maximum Q-value and the joint action of the other agents that results in this Q-value
        q_all = critic.forward(batch, compute_all=True)[0][:, :-1]
        
        # Original: q_all = q_all.max(dim=3)[0]
        # Adaptive: blend max and mean
        alpha = self.scheduler.get_alpha()
        
        q_max = q_all.max(dim=3)[0]
        q_mean = q_all.mean(dim=3) # Mean over a_{-i}
        
        q_adaptive = alpha * q_max + (1 - alpha) * q_mean
        
        q_selected = th.gather(q_adaptive, -1, actions).squeeze(-1)
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
