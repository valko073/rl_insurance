import gym
from gym import spaces
import numpy as np

LEN_LOOKBACK = 10
LEN_EPISODE = 500



class InsuranceEnv(gym.Env):

    def __init__(self, num_agents, num_insurances):
        super(InsuranceEnv, self).__init__()

        self.NUM_AGENTS = num_agents
        self.NUM_INSURANCES = num_insurances

        self.safe_mu = 1
        self.safe_sigma = 0.1
        self.risky_mu = 1
        self.risky_sigma = 1
        self.insurance_return = 0.5

        self.action_space = spaces.Discrete(2+2*self.NUM_INSURANCES)
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
        }

        for i in range(self.NUM_INSURANCES*2):
            if i % 2 == 0:
                self.action_switcher[i+2] = (self.safe_mu, self.safe_sigma, 1.)
            else:
                self.action_switcher[i+2] = (self.risky_mu, self.risky_sigma, 1.)


        self.observation_space = spaces.Dict({
            'was_insured': spaces.MultiBinary(LEN_LOOKBACK),
            'insurance_costs': spaces.Box(low=0.0, high=1.0, shape=(LEN_LOOKBACK,), dtype=np.float32),
            'new_cost': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        self.reset()

    def step(self, agent_actions):
        agent_rewards = [0.0 for _ in range(self.NUM_AGENTS)]
        insurance_rewards = [-.01 for _ in range(self.NUM_INSURANCES)]

        if not type(agent_actions) == list or type(agent_actions) == np.array:
            agent_actions = [agent_actions]

        for agent_id, agent_action in enumerate(agent_actions):

            mu, sigma, insured = self.action_switcher.get(agent_action, (0, 0, 0.0))

            insurance_id = agent_action//2 - 1
            insurance_cost = self.current_cost[insurance_id]

            self.action_counter[agent_id, agent_action] += 1

            agent_reward = np.random.normal(mu, sigma)
            insurer_reward = 0
            if insured == 1.:
                if agent_reward < self.insurance_return:
                    insurer_reward = agent_reward - self.insurance_return + insurance_cost
                    agent_reward = self.insurance_return - insurance_cost
                else:
                    agent_reward -= insurance_cost
                    insurer_reward = insurance_cost

            self.was_insured[agent_id].pop(0)
            self.was_insured[agent_id].append(insured)
            self.insurance_costs[insurance_id].pop(0)
            self.insurance_costs[insurance_id].append(self.get_insurance_cost(insurance_id))

            agent_rewards[agent_id] = agent_reward
            insurance_rewards[insurance_id] += insurer_reward

        self.num_trials -= 1

        done = self.num_trials == 0

        observation = self.get_obs()

        """Try to force insurance to make price competitive"""
        # insurer_reward -= .01

        return [observation for _ in range(self.NUM_INSURANCES+self.NUM_AGENTS)], (*insurance_rewards, *agent_rewards),\
               done, None

        """Get action from insurance"""

        """Get action from agent"""

        """"Calculate rewards"""

        """Save rewards in replay memory"""

        """Train insurance and agent"""

    def get_obs(self):
        obs = np.zeros((self.NUM_INSURANCES, 21), dtype=np.float32)

        obs[:, :10] = self.was_insured
        obs[:, 10:20] = self.insurance_costs
        obs[:, 20] = self.current_cost
        return obs

    def reset(self):
        self.was_insured = [[0.0 for _ in range(LEN_LOOKBACK)] for _ in range(self.NUM_AGENTS)]
        self.insurance_costs = [[0.0 for _ in range(LEN_LOOKBACK)] for _ in range(self.NUM_INSURANCES)]
        self.set_insurance_cost()
        self.num_trials = LEN_EPISODE

        self.action_counter = np.zeros((self.NUM_AGENTS, self.NUM_INSURANCES*2+2))

        return [self.get_obs() for _ in range(self.NUM_INSURANCES+self.NUM_AGENTS)]

    def set_insurance_cost(self, insurance_cost: float = 0.0, insurance_id=None):
        if insurance_id is None:
            self.current_cost = [insurance_cost for _ in range(self.NUM_INSURANCES)]
        else:
            if insurance_id > self.NUM_INSURANCES:
                raise Exception('Insurance ID higher than number of insurances (ID: {}, insurances: {}'.format(
                    insurance_id, self.NUM_INSURANCES
                ))
            self.current_cost[insurance_id] = insurance_cost

    def get_insurance_cost(self, insurance_id=None):
        if insurance_id is None:
            return self.current_cost
        else:
            if insurance_id > self.NUM_INSURANCES:
                raise Exception('Insurance ID higher than number of insurances (ID: {}, insurances: {}'.format(
                    insurance_id, self.NUM_INSURANCES
                ))
            return self.current_cost[insurance_id]

    def render(self, mode='human', close=False):
        pass
