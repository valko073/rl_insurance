import gym
from gym import spaces
import numpy as np

LEN_LOOKBACK = 10
LEN_EPISODE = 500
NUM_INSURANCES = 1
NUM_AGENTS = 1


class InsuranceEnv(gym.Env):

    def __init__(self):
        super(InsuranceEnv, self).__init__()

        self.safe_mu = 1
        self.safe_sigma = 0.1
        self.risky_mu = 1
        self.risky_sigma = 1
        self.insurance_return = 0.5

        self.action_space = spaces.Discrete(2+2*NUM_INSURANCES)
        """
        0: Safe non-insured
        1: Risky non-insured
        2: Safe insured
        3: Risky insured
        4: Safe insured2
        ...
        """
        self.action_switcher = {
            0: (self.safe_mu, self.safe_sigma, 0.),
            1: (self.risky_mu, self.risky_sigma, 0.),
            2: (self.safe_mu, self.safe_sigma, 1.),
            3: (self.risky_mu, self.risky_sigma, 1.)
        }

        self.observation_space = spaces.Dict({
            'was_insured': spaces.MultiBinary(LEN_LOOKBACK),
            'insurance_costs': spaces.Box(low=0.0, high=1.0, shape=(LEN_LOOKBACK,), dtype=np.float32),
            'new_cost': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        self.reset()

    def step(self, agent_actions: list[int]):
        for agent_action in agent_actions:
            mu, sigma, insured = self.action_switcher.get(agent_action, (0, 0, 0.0))

            self.action_counter[agent_action] += 1

            agent_reward = np.random.normal(mu, sigma)
            insurer_reward = -.01
            if insured == 1.:
                if agent_reward < self.insurance_return:
                    # print("agent reward:", agent_reward)
                    # print("agent action:", agent_action)
                    # print("insurance_return:", self.insurance_return)
                    # print("current cost:", self.current_cost)
                    # print("insurer reward:", insurer_reward)
                    # print("agent reward:", agent_reward)
                    # print(" ")
                    insurer_reward = agent_reward - self.insurance_return + self.current_cost
                    agent_reward = self.insurance_return - self.current_cost
                else:
                    agent_reward -= self.current_cost
                    insurer_reward = self.current_cost

            self.num_trials -= 1

            done = self.num_trials == 0

            self.was_insured.pop(0)
            self.was_insured.append(insured)
            self.insurance_costs.pop(0)
            self.insurance_costs.append(self.get_insurance_cost())

            observation = self.get_obs()

        """Try to force insurance to make price competitive"""
        # insurer_reward -= .01

        return (observation, observation), (insurer_reward, agent_reward), done, None

        """Get action from insurance"""

        """Get action from agent"""

        """"Calculate rewards"""

        """Save rewards in replay memory"""

        """Train insurance and agent"""

    def get_obs(self):
        # return {
        #     'was insured': np.array(self.was_insured, dtype=int),
        #     'insurance_costs': np.array(self.insurance_costs, dtype=np.float32),
        #     'new_cost': np.array(self.current_cost, dtype=np.float32)
        # }
        # obs = np.append(self.was_insured, self.insurance_costs.append(self.current_cost))
        obs = np.zeros(21, dtype=np.float32)
        obs[:10] = self.was_insured
        obs[10:20] = self.insurance_costs
        obs[20] = self.current_cost

        return obs

    def reset(self):
        self.was_insured = [0.0 for _ in range(LEN_LOOKBACK)]
        self.insurance_costs = [0.0 for _ in range(LEN_LOOKBACK)]
        self.set_insurance_cost()
        self.num_trials = LEN_EPISODE

        self.action_counter = [0, 0, 0, 0]

        return [self.get_obs(), self.get_obs()]

    def set_insurance_cost(self, insurance_cost: float = 0.0):
        # assert insurance_cost >= 0.0, "insurance cost must be positive or zero"
        self.current_cost = insurance_cost

    def get_insurance_cost(self):
        return self.current_cost

    def render(self, mode='human', close=False):
        pass
