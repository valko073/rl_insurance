from rl.agents import DDPGAgent
import numpy as np


class CustomDDPGAgent(DDPGAgent):
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):
        super(CustomDDPGAgent, self).__init__(nb_actions, actor, critic, critic_action_input, memory,
                 gamma, batch_size, nb_steps_warmup_critic, nb_steps_warmup_actor,
                 train_interval, memory_interval, delta_range, delta_clip,
                random_process, custom_model_objects, target_model_update, **kwargs)

    def select_action(self, state):




        # batch = self.process_state_batch([state])
        # print(batch)
        # print(type(batch))
        # print(batch.shape)

        # batch = np.array([np.array([s]) for s in list(state[0].values())])
        # batch = state[0].reshape(1,1,21)
        batch = state[0]

        # action = self.actor.predict_on_batch(batch).flatten()
        action = self.actor.predict(batch, verbose=0).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        # print(self.actor.summary())
        #print(action)

        return action
