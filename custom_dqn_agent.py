from rl.agents import DQNAgent
import numpy as np
import copy

class CustomDQNAgent(DQNAgent):
    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=False, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(CustomDQNAgent, self).__init__(model, policy, test_policy, enable_double_dqn, enable_dueling_network,
                                             dueling_type, *args, **kwargs)
        self.to_restart = copy.deepcopy(policy)

    def restart_policy(self):
        self.policy = self.to_restart


