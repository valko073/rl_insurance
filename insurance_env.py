import gym
from gym import spaces
import numpy as np

LEN_LOOKBACK = 10
LEN_EPISODE = 50


class InsuranceEnv(gym.Env):

    def __init__(self, agents, insurances):
        super(InsuranceEnv, self).__init__()

        self.safe_mu = 1
        self.safe_sigma = 0.1
        self.risky_mu = 1
        self.risky_sigma = 1
        self.insurance_return = 0.5

        self.action_space = spaces.Discrete(2+2*len(insurances))
        """
        0: Safe non-insured
        1: Risky non-insured
        2: Safe insured
        3: Risky insured
        4: Safe insured2
        ...
        """
        self.action_switcher = {
            0: (self.safe_mu, self.safe_sigma, False),
            1: (self.risky_mu, self.risky_sigma, False),
            2: (self.safe_mu, self.safe_sigma, True),
            3: (self.risky_mu, self.risky_sigma, True)
        }

        self.observation_space = spaces.Dict({
            'was_insured': spaces.MultiBinary(LEN_LOOKBACK),
            'insurance_costs': spaces.Box(low=0.0, high=1.0, shape=(LEN_LOOKBACK,), dtype=np.float32),
            'new_cost': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

    def step(self, action: int):
        mu, sigma, insured = self.action_switcher.get(action, (0, 0, False))

        agent_reward = np.random.normal(mu, sigma)
        insurer_reward = 0.0
        if insured:
            if agent_reward < self.insurance_return:
                insurer_reward = agent_reward - self.insurance_return + self.insurance_cost
                agent_reward = self.insurance_return - self.insurance_cost

        self.num_trials -= 1

        done = self.num_trials == 0

        observation = self._get_obs()

        return observation, agent_reward, insurer_reward, done

        """Get action from insurance"""

        """Get action from agent"""

        """"Calculate rewards"""

        """Save rewards in replay memory"""

        """Train insurance and agent"""

    def _get_obs(self):
        return {
            'was insured': np.array(self.was_insured, dtype=int),
            'insurance_costs': np.array(self.insurance_costs, dtype=np.float32),
            'new_cost': np.array(self.current_cost, dtype=np.float32)
        }

    def reset(self):
        self.was_insured = [0 for _ in range(LEN_LOOKBACK)]
        self.insurance_costs = [0.0 for _ in range(LEN_LOOKBACK)]
        self.current_cost = 0.0
        self.num_trials = LEN_EPISODE

    def set_insurance_cost(self, insurance_cost: float = 0.0):
        self.current_cost = insurance_cost

    def get_insurance_cost(self):
        return self.current_cost

    def render(self, mode='human', close=False):
        pass
