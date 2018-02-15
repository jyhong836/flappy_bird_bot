import json
import argparse
import random
import numpy as np
from collections import deque
from ple.games.flappybird import FlappyBird

from fb_trainer import Trainer
from fbmodels import ModelFactory

# The file borrows many codes from 'https://github.com/keon/deep-q-learning'

class AgentType(object):
    def __init__(self):
        pass

    def act(self, reward, obs):
        pass

    def setup(self, state_size, action_space):
        pass

class MyAgent(AgentType):
    """My agent

        A simple agent playing FlappyBird.
    """

    def __init__(self, 
        state_size    = 0,
        action_space  = [],
        batch_size    = 32,
        memory_size   = 50000,
        gamma         = 0.99,
        init_epsilon  = 1.0,
        epsilon_min   = 0.0001,
        learning_rate = 0.001,
        n_observation = 10000,
        n_explore     = 100000):
        # print(action_space)
        # assert len(state_size) == 1
        self.state_size   = state_size
        self.action_space = action_space
        self.batch_size   = batch_size

        self.memory = deque(maxlen=memory_size)

        self.gamma         = gamma    # discount rate
        self.init_epsilon  = init_epsilon
        self.epsilon       = init_epsilon  # exploration rate
        self.epsilon_min   = epsilon_min
        # self.epsilon_decay = 0.99
        self.learning_rate = learning_rate
        self.n_observation = n_observation  # timesteps to observe before training
        self.n_explore     = n_explore  # frames over which to anneal epsilon

        # private
        self._state = None
    
    def setup(self, state_size, action_space):
        self.state_size   = state_size
        self.action_space = action_space

        # init model
        self.model = self._build_model()
    
    def _build_model(self):
        model = ModelFactory.naive_dqn_v1(self.state_size, len(
            self.action_space), self.learning_rate)
        return model

    def act(self, reward, state):
        self._state = state
        if np.random.rand() <= self.epsilon:
            return self.action_space[np.random.randint(0, len(self.action_space))]
        else:
            action_prob = self.model.predict(self._state_preprocessor(state))
            return self._idx2act(np.argmax(action_prob))

    def study(self):
        """Learn models from memory or by replaying.
        """
        if len(self.memory) < self.batch_size or len(self.memory) < self.n_observation:
            return # not study

        def do_with_play(curstate, action, reward, next_state, game_over):
            # print(curstate, action, reward, next_state, game_over)
            target = self.model.predict(curstate)
            if game_over:
                target[0][action] = reward
            else:
                # TODO replace the 'self.model' with a stable one.
                target[0][action] = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            self.model.fit(curstate, target, epochs=1, verbose=0)
            return None

        self._replay(self.batch_size, do_with_play)
        if self.epsilon > self.epsilon_min and len(self.memory) > self.n_observation:
            # self.epsilon *= self.epsilon_decay
            self.epsilon -= (self.init_epsilon - self.epsilon_min) / self.n_explore

    def _replay(self, batch_size, do_with_play):
        """

        do_with_play (curstate, action, reward, state, game_over)-> ??
        """
        minibatch = random.sample(self.memory, batch_size)
        for play_tuple in minibatch:
            do_with_play(*play_tuple)

    def _state_preprocessor(self, state):
        return np.reshape(state, [1, self.state_size[0]])

    def _act2idx(self, action):
        return self.action_space.index(action)

    def _idx2act(self, index):
        return self.action_space[index]

    def remember(self, action, reward, next_state, game_over):
        """Remember current information.

        'memory' will store info formatted as:
            (curstate, action, reward, state, game_over)
        """
        # print(state)
        curstate = self._state
        # self._state = self._state_preprocessor(next_state)
        if curstate is None:
            print('Init state is not set! Call set_state() at each episode.')
            raise ValueError
        self.memory.append(
            (self._state_preprocessor(curstate),
            self._act2idx(action), 
            reward, 
            self._state_preprocessor(next_state), 
            game_over))
    
    def load(self, file_name):
        self.model.load_weights(file_name)

    def save(self, file_name):
        if len(self.memory) < self.batch_size or len(self.memory) < self.n_observation:
            print('Model is not trained yet, not save.')
            return # not trained
        self.model.save_weights(file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a robot for "Flappy Bird" game.')
    parser.add_argument('-d', '--save-dir', dest='save_folder', default=None, type=str,
                        help='Select a folder to store the saved data e.g., network.')
    # parser.add_argument('-ns', '--no-display-screen', dest='not_display_screen',
    #                     default=False, help='Display game to the screen.', action='store_true')
    parser.add_argument('-l', '--load', type=str, default=None, dest='load_file', help='Select file under "save-folder" to load.')
    # parser.add_argument('-e', '--episode', dest='n_episodes', default=200000, type=int, help='Number of episodes to run.')
    # parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-f', '--conf-file', help='Select config file',
                        default='conf.json', type=str, dest='conf_file')
    # parser.add_argument('-fq', '--save-freq', help='Save frequency',
    #                     default=0, type=int, dest='save_freq')
    args = parser.parse_args()

    # print(args.not_display_screen)

    # predefined parameters
    # n_episodes = 2

    game = FlappyBird()

    with open(args.conf_file) as cf:
        loaded_config = json.load(cf)
        agent = MyAgent(**(loaded_config['agent']))
        trainer = Trainer(game, agent, **(loaded_config['trainer']))
        trainer.load(args.load_file)
        trainer.train()
        # trainer.play()
        trainer.save()
        trainer.save_screen()  # for testing screen

