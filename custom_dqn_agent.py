from rl.agents import DQNAgent
import numpy as np


class CustomDQNAgent(DQNAgent):
    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=False, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(CustomDQNAgent, self).__init__(model, policy, test_policy, enable_double_dqn, enable_dueling_network,
                                             dueling_type, *args, **kwargs)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.mu_model.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        print(action)

        return action

    def forward(self, observation):
        action = super().forward(observation)
        print(action)
        return action
