import gym
from gym import spaces


class InsuranceEnv(gym.Env):

    def __init__(self, agents, insurances):
        self.insurance_cost = 0.0
        self.num_trials = 50

        super(InsuranceEnv, self).__init__()

        self.action_space = spaces.Discrete(2+2*len(insurances))
        """
        0: Safe non-insured
        1: Risky non-insured
        2: Safe insured
        3: Risky insured
        4: Safe insured2
        ...
        """

        self.observation_space = spaces.Tuple(spaces.Box(), spaces.Discrete(10))

    def step(self, action):
        return

    def reset(self):
        return

    def set_insurance_cost(self, insurance_cost: float):
        pass

    def get_insurance_cost(self):
        return self.insurance_cost


    def render(self, mode='human', close=False):
        pass
