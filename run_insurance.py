import numpy as np
import gym
from copy import deepcopy
from logging import getLogger

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.policy import BoltzmannQPolicy
from rl.processors import MultiInputProcessor

from insurance_env import InsuranceEnv

from custom_ddpg_agent import CustomDDPGAgent


np.random.seed(123)
NUM_HIDDEN_UNITS = 32
logger = getLogger()




def fit_n_agents(env, nb_steps, agents=None, nb_max_episode_steps=None, logger=None, log_dir=None):
    for agent in agents:
        if not agent.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet.'
                ' Please call `compile()` before `fit()`.')

        agent.training = True
        agent._on_train_begin()

    episode = 0
    observations = [None for _ in agents]
    episode_rewards = [None for _ in agents]
    episode_steps = [None for _ in agents]

    for agent in agents:
        agent.step = 0
    did_abort = False
    try:
        while agents[0].step < nb_steps:
            if observations[0] is None:  # start of a new episode
                observations = deepcopy(env.reset())
                for i, agent in enumerate(agents):
                    episode_steps[i] = 0
                    episode_rewards[i] = 0.
                    # Obtain the initial observation by resetting the environment.
                    agent.reset_states()
                    if agent.processor is not None:
                        observations[i] = agent.processor.process_observation(observations[i])
                    assert observations[i] is not None
                    # At this point, we expect to be fully initialized.
                    assert episode_rewards[i] is not None
                    assert episode_steps[i] is not None
                    assert observations[i] is not None

            actions = []
            for i, agent in enumerate(agents):
                # Run a single step.
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                print("i: ",i,observations[i])
                actions.append(agent.forward(observations[i]))
                if agent.processor is not None:
                    actions[i] = agent.processor.process_action(actions[i])

            accumulated_info = {}
            done = False

            env.step_i = agents[0].step

            env.set_insurance_cost(actions[0])

            observations, r, done, info = env.step(actions[1])
            print(observations)

            observations = deepcopy(observations)
            print(observations)

            for i, agent in enumerate(agents):
                if agent.processor is not None:
                    observations[i], r[i], done, info = agent.processor.process_step(observations[i], r[i], done, info)

            if nb_max_episode_steps and episode_steps[0] >= nb_max_episode_steps - 1:
                # Force a terminal state.
                done = True

            for i, agent in enumerate(agents):
                metrics = agent.backward(r[i], terminal=done)
                episode_rewards[i] += r[i]
                episode_steps[i] += 1
                agent.step += 1

            if done:
                for i, agent in enumerate(agents):
                    agent.forward(observations[i])
                    agent.backward(0., terminal=False)

                logger.write_log('episode_return', np.sum(episode_rewards), episode)
                logger.write_log('bargaining_succes', info['bargaining_succes'], episode)
                for i, agent in enumerate(agents):
                    logger.write_log('episode_return_agent-{}'.format(i), r[i], episode)
                # for key, value in info.items():
                #    logger.write_log(key, value, agents[0].step)

                observations = [None for _ in agents]
                episode_steps = [None for _ in agents]
                episode_rewards = [None for _ in agents]
                episode += 1

                print("DONE")

            print("step: ",env.step_i)

    except KeyboardInterrupt:
        # We catch keyboard interrupts here so that training can be be safely aborted.
        # This is so common that we've built this right into this function, which ensures that
        # the `on_train_end` method is properly called.
        did_abort = True
    for i, agent in enumerate(agents):
        agent._on_train_end()


if __name__ == '__main__':
    env = InsuranceEnv()
    agents = []

    print(env.observation_space.shape)

    agent_model = Sequential()
    agent_model.add(Flatten(input_shape=(1,) + (21,)))
    agent_model.add(Dense(NUM_HIDDEN_UNITS))
    agent_model.add(Activation('relu'))
    agent_model.add(Dense(NUM_HIDDEN_UNITS))
    agent_model.add(Activation('relu'))
    agent_model.add(Dense(NUM_HIDDEN_UNITS))
    agent_model.add(Activation('relu'))
    agent_model.add(Dense(env.action_space.n))
    agent_model.add(Activation('linear'))
    print(agent_model.summary())

    ins_actor = Sequential()
    ins_actor.add(Flatten(input_shape=(1,) + (21,)))
    ins_actor.add(Dense(NUM_HIDDEN_UNITS))
    ins_actor.add(Activation('relu'))
    ins_actor.add(Dense(NUM_HIDDEN_UNITS))
    ins_actor.add(Activation('relu'))
    ins_actor.add(Dense(NUM_HIDDEN_UNITS))
    ins_actor.add(Activation('relu'))
    ins_actor.add(Dense(1))
    ins_actor.add(Activation('sigmoid'))
    print(ins_actor.summary())

    action_input = Input(shape=(1,), name='action_input')
    observation_input = Input(shape=(1,) + (21,), name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(NUM_HIDDEN_UNITS)(x)
    x = Activation('relu')(x)
    x = Dense(NUM_HIDDEN_UNITS)(x)
    x = Activation('relu')(x)
    x = Dense(NUM_HIDDEN_UNITS)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    ins_critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(ins_critic.summary(()))

    ag_memory = SequentialMemory(limit=10000, window_length=1)
    ag_policy = BoltzmannQPolicy()
    ag_dqn = DQNAgent(model=agent_model, nb_actions=env.action_space.n, memory=ag_memory, nb_steps_warmup=100,
                      target_model_update=1e-2, policy=ag_policy)
    ag_dqn.compile(Adam(lr=.001), metrics=['mae'])

    ins_memory = SequentialMemory(limit=10000, window_length=1)
    ins_random_process = OrnsteinUhlenbeckProcess(size=1, theta=.15, mu=0, sigma=.3)
    ins_agent = CustomDDPGAgent(nb_actions=1, actor=ins_actor, critic=ins_critic, critic_action_input=action_input,
                          memory=ins_memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=ins_random_process, gamma=.99, target_model_update=1e-3)
    # ins_agent.processor = MultiInputProcessor(3)
    ins_agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    agents.append(ins_agent)
    agents.append(ag_dqn)

    fit_n_agents(env=env, nb_steps=100, agents=agents, nb_max_episode_steps=100, logger=logger)