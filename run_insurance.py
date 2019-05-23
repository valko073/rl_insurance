import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.memory import EpisodeParameterMemory, SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from insurance_env import InsuranceEnv

np.random.seed(123)
NUM_HIDDEN_UNITS = 32

env = InsuranceEnv()


agent_model = Sequential()
agent_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
agent_model.add(Dense(NUM_HIDDEN_UNITS))
agent_model.add(Activation('relu'))
agent_model.add(Dense(NUM_HIDDEN_UNITS))
agent_model.add(Activation('relu'))
agent_model.add(Dense(NUM_HIDDEN_UNITS))
agent_model.add(Activation('relu'))
agent_model.add(Dense(env.action_space.n))
agent_model.add(Activation('linear'))
print(agent_model.summary())

insurance_actor = Sequential()
insurance_actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
insurance_actor.add(Dense(NUM_HIDDEN_UNITS))
insurance_actor.add(Activation('relu'))
insurance_actor.add(Dense(NUM_HIDDEN_UNITS))
insurance_actor.add(Activation('relu'))
insurance_actor.add(Dense(NUM_HIDDEN_UNITS))
insurance_actor.add(Activation('relu'))
insurance_actor.add(Dense(1))
insurance_actor.add(Activation('linear'))
print(insurance_actor.summary())

action_input = Input(shape=(1,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(NUM_HIDDEN_UNITS)(x)
x = Activation('relu')(x)
x = Dense(NUM_HIDDEN_UNITS)(x)
x = Activation('relu')(x)
x = Dense(NUM_HIDDEN_UNITS)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
insurance_critic = Model(inputs=[action_input, observation_input], outputs=x)
print(insurance_critic.summary(()))
