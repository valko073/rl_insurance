import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import numpy as np
import gym
from copy import deepcopy
from logging import getLogger
import argparse
from comet_ml import Experiment
import neptune

from datetime import datetime
import pickle

from util import generate_agent_model, generate_insurance_model
import util

from insurance_env import InsuranceEnv, LEN_EPISODE
from config import EnvConfig



seed = np.random()
np.random.seed(seed)

logger = getLogger()
comet_cfg = EnvConfig()


def fit_n_agents(env, nb_steps, agents=None, num_agents=1,  num_insurances=1, nb_max_episode_steps=None, logger=None, log_dir=None):
    print('NUM_AGENTS:', len(agents))
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

    to_log = []
    try:
        while agents[0].step < nb_steps:
            insurance_costs = [[] for _ in range(num_insurances)]
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
                actions.append(agent.forward(observations[i]))
                if agent.processor is not None:
                    actions[i] = agent.processor.process_action(actions[i])

            accumulated_info = {}
            done = False

            env.step_i = agents[0].step

            for i in range(num_insurances):
                env.set_insurance_cost(actions[i][0],i)
                insurance_costs[i].append(actions[i][0])

            observations, r, done, info = env.step(actions[num_insurances:])

            observations = deepcopy(observations)

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

            to_log.append([observations[0], actions, r])


            if args.comet:
                for i in range(num_insurances):
                    experiment.log_metric("insurance_cost_"+str(i), actions[i][0])
                    neptune.send_metric('insurance_cost_'+str(i), actions[i][0])

            if done:
                if args.comet:
                    # experiment.log_metrics({"num_safe_non_insured": env.action_counter[0],
                    #                         "num_risky_non_insured": env.action_counter[1],
                    #                         "num_safe_insured": env.action_counter[2],
                    #                         "num_risky_insured": env.action_counter[3],
                    #                         "avg_insurance_cost": np.mean(insurance_costs),
                    #                         "num_safe": env.action_counter[0]+env.action_counter[2],
                    #                         "num_risky": env.action_counter[1]+env.action_counter[3],
                    #                         "num_insured": env.action_counter[2]+env.action_counter[3],
                    #                         "num_non_insured": env.action_counter[0]+env.action_counter[1]
                    #                         })

                    for (agent_id, i), action_count in np.ndenumerate(env.action_counter):
                        if i < 2:
                            if i % 2 == 0:
                                neptune.send_metric("num_safe_non_insured_agent_"+str(agent_id), action_count)
                            else:
                                neptune.send_metric("num_risky_non_insured_agent_"+str(agent_id), action_count)
                        else:
                            if i % 2 == 0:
                                if agent_id == 0:
                                    neptune.send_metric("num_safe_insured_"+str(i//2-1), np.sum(env.action_counter[:, i]))
                                neptune.send_metric("num_safe_insured_agent_" + str(agent_id) +
                                                    "_insurance_"+str(i//2-1), action_count)
                            else:
                                if agent_id == 0:
                                    neptune.send_metric("num_risky_insured_"+str(i//2-1), np.sum(env.action_counter[:, i]))
                                neptune.send_metric("num_risky_insured_agent_"+str(agent_id) +
                                                    "_insurance_"+str(i//2-1), action_count)

                    neptune.send_metric("avg_insurance_cost", np.mean(insurance_costs))
                    neptune.send_metric("avg_insurance_cost_scaled", np.mean(insurance_costs)*LEN_EPISODE)
                    neptune.send_metric("num_safe", np.sum(env.action_counter[:, 0::2]))
                    neptune.send_metric("num_risky", np.sum(env.action_counter[:, 1::2]))
                    neptune.send_metric("num_insured", np.sum(env.action_counter[:, 2:]))
                    neptune.send_metric("num_non_insured", np.sum(env.action_counter[:, :2]))

                    neptune.send_metric("num_safe_non_insured", np.sum(env.action_counter[:, 0]))
                    neptune.send_metric("num_risky_non_insured", np.sum(env.action_counter[:, 1]))

                    for i, v in enumerate(np.mean(insurance_costs, axis=1)):
                        neptune.send_metric("avg_insurance_cost_"+str(i), v)
                        neptune.send_metric("avg_insurance_cost_scaled_" + str(i), v*LEN_EPISODE)

                    for i in range(num_insurances):
                        neptune.send_metric("num_insured_"+str(i), np.sum(env.action_counter[:, i*2+2:i*2+4]))

                    experiment.set_step(env.step_i)

                for i, agent in enumerate(agents):
                    agent.forward(observations[i])
                    agent.backward(0., terminal=False)

                # logger.info('episode_return', np.sum(episode_rewards), episode)
                # logger.info('bargaining_succes', info['bargaining_succes'], episode)
                print('episode_return', np.sum(episode_rewards), episode)
                for i, agent in enumerate(agents):
                    logger.info('episode_return_agent-{}'.format(i), r[i], episode)
                    print('episode_return_agent-{}'.format(i), r[i], episode)
                    if i < num_insurances:
                        model_type = "insurance_"+str(i)
                    else:
                        model_type = "agent_"+str(i-num_insurances)
                    if args.comet:
                        experiment.log_metric("reward_"+model_type, np.sum(episode_rewards[i]))
                        neptune.send_metric("reward_"+model_type, np.sum(episode_rewards[i]))
                # for key, value in info.items():
                #    logger.write_log(key, value, agents[0].step)

                observations = [None for _ in agents]
                episode_steps = [None for _ in agents]
                episode_rewards = [None for _ in agents]
                episode += 1



            # print("step: ", env.step_i)

    # except KeyboardInterrupt:
    finally:
        # We catch keyboard interrupts here so that training can be be safely aborted.
        # This is so common that we've built this right into this function, which ensures that
        # the `on_train_end` method is properly called.
        did_abort = True
        with open('logs/outfile-%s.p' % datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), 'wb') as fp:
            pickle.dump(to_log, fp)
        for i, agent in enumerate(agents):
            if i < num_insurances:
                model_type = "insurance_"+str(i)
            else:
                model_type = "agent_"+str(i-num_insurances)
            filename = 'models/'+model_type+'-%s.h5' % datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            agent.save_weights(filename)
            agent._on_train_end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--num_insurances", type=int, default=1)

    args = parser.parse_args()

    env = InsuranceEnv(args.num_agents, args.num_insurances)
    agents = []

    for i in range(args.num_insurances):
        agents.append(generate_insurance_model(env))
    for i in range(args.num_agents):
        agents.append(generate_agent_model(env))

    if args.comet:
        experiment = Experiment(api_key=comet_cfg.comet_api_key,
                                project_name=comet_cfg.comet_project_name, workspace=comet_cfg.comet_workspace)

        neptune.init(api_token=comet_cfg.neptune_token, project_qualified_name=comet_cfg.neptune_project_name)
        neptune.create_experiment()
        neptune.set_property('num_insurances', args.num_insurances)
        neptune.set_property('num_agents', args.num_agents)
        neptune.set_property('memory_limit', util.MEMORY_LIMIT)
        neptune.set_property('target_model_update', util.TARGET_MODEL_UPDATE)
        neptune.set_property('num_steps', args.num_steps)
        neptune.set_property('risky_mu', env.risky_mu)
        neptune.set_property('safe_mu', env.safe_mu)
        neptune.set_property('insurance_return', env.insurance_return)
        neptune.set_property('ag_model_eps_final', agents[-1].policy.value_min)
        neptune.set_property('random_seed', seed)

    fit_n_agents(env=env, nb_steps=args.num_steps, agents=agents, num_agents=args.num_agents,
                 num_insurances=args.num_insurances, nb_max_episode_steps=1000, logger=logger)
    print('done')
    if args.comet:
        neptune.stop()
