from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.processors import MultiInputProcessor
import numpy as np

from custom_ddpg_agent import CustomDDPGAgent
from custom_dqn_agent import CustomDQNAgent

# np.random.seed(123)
NUM_HIDDEN_UNITS = 32
MEMORY_LIMIT = 100
TARGET_MODEL_UPDATE = .09


def generate_agent_model(env=None):
    agent_model = Sequential()
    agent_model.add(Flatten(input_shape=(1,) + (env.NUM_INSURANCES,21)))
    agent_model.add(Dense(NUM_HIDDEN_UNITS))
    agent_model.add(Activation('relu'))
    agent_model.add(Dense(NUM_HIDDEN_UNITS))
    agent_model.add(Activation('relu'))
    agent_model.add(Dense(NUM_HIDDEN_UNITS))
    agent_model.add(Activation('relu'))
    agent_model.add(Dense(env.action_space.n))
    agent_model.add(Activation('linear'))
    # print(agent_model.summary())

    ag_memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=1)
    # ag_policy = BoltzmannQPolicy()
    ag_policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=.95, value_min=0, value_test=0,
                                     nb_steps=5000)
    ag_dqn = DQNAgent(model=agent_model, nb_actions=env.action_space.n, memory=ag_memory, nb_steps_warmup=100,
                      target_model_update=TARGET_MODEL_UPDATE, policy=ag_policy)
    ag_dqn.compile(Adam(lr=.0001), metrics=['mae'])

    print(type(ag_dqn))

    return ag_dqn


def generate_insurance_model(env=None):
    ins_actor = Sequential()
    ins_actor.add(Flatten(input_shape=(1,) + (env.NUM_INSURANCES, 21)))
    ins_actor.add(Dense(NUM_HIDDEN_UNITS))
    ins_actor.add(Activation('relu'))
    ins_actor.add(Dense(NUM_HIDDEN_UNITS))
    ins_actor.add(Activation('relu'))
    ins_actor.add(Dense(NUM_HIDDEN_UNITS))
    ins_actor.add(Activation('relu'))
    ins_actor.add(Dense(1))
    ins_actor.add(Activation('softsign'))
    # print(ins_actor.summary())
    # print(ins_actor.layers[-1].activation)

    action_input = Input(shape=(1,), name='action_input')
    observation_input = Input(shape=(1,) + (env.NUM_INSURANCES,21), name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(NUM_HIDDEN_UNITS)(x)
    x = Activation('relu')(x)
    x = Dense(NUM_HIDDEN_UNITS)(x)
    x = Activation('relu')(x)
    x = Dense(NUM_HIDDEN_UNITS)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('softsign')(x)
    ins_critic = Model(inputs=[action_input, observation_input], outputs=x)
    # print(ins_critic.summary(()))

    ins_memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=1)
    # ins_random_process = OrnsteinUhlenbeckProcess(size=1, theta=.15, mu=0, sigma=.3)
    ins_random_process = GaussianWhiteNoiseProcess(mu=0, sigma=0.2, sigma_min=0.005, n_steps_annealing=5000)
    # ins_random_process = None
    ins_agent = DDPGAgent(nb_actions=1, actor=ins_actor, critic=ins_critic, critic_action_input=action_input,
                          memory=ins_memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=ins_random_process, gamma=.99, target_model_update=TARGET_MODEL_UPDATE)
    # ins_agent.processor = MultiInputProcessor(3)
    ins_agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=['mae'])

    print(type(ins_agent))

    return ins_agent
