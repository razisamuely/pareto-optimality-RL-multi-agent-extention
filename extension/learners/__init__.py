"""
PAC Extension Learners

All variants inherit from base PACActorCriticLearner and override
only the Q-value computation methods.
"""

import sys
import os

# Add parent directory to path to import from EPyMARL
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from learners.actor_critic_pac_learner import PACActorCriticLearner

__all__ = ['PACActorCriticLearner']
