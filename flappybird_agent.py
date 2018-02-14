import random
import numpy as np
from collections import deque
from ple.games.flappybird import FlappyBird

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend

from fb_trainer import Trainer

class MyModelFact():
    """My model factories
    """
    
    @staticmethod
    def naive_dqn(state_size, action_size, learning_rate):
        """ Neural Net for Deep-Q learning Model
        """
        print('Build Deep Q-learnig Model.')

        def huber_loss(target, prediction):
            """sqrt(1+error^2)-1"""
            return keras.backend.mean(keras.backend.sqrt(1+keras.backend.square(prediction - target))-1, axis=-1)

        model = Sequential()
        # print('### '+repr(state_size))
        model.add(Dense(24, input_shape=state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=learning_rate))
        return model

class AgentType(object):
    def __init__(self, state_size, action_space):
        pass

    def act(self, reward, obs):
        pass

class MyAgent(AgentType):
    """My agent

        A simple agent playing FlappyBird.
    """

    def __init__(self, state_size, action_space, batch_size=10):
        # assert len(state_size) == 1
        self.state_size   = state_size
        self.action_space = action_space
        self.batch_size   = batch_size

        self.memory = deque(maxlen=2000)

        self.gamma         = 0.95    # discount rate
        self.epsilon       = 1.0  # exploration rate
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001

        # private
        self._state = None

        # init model
        self.model = self._build_model()
    
    def _build_model(self):
        model = MyModelFact.naive_dqn(self.state_size, len(self.action_space), self.learning_rate)
        return model

    def set_state(self, state):
        self._state = self._state_preprocessor(state)

    def act(self, reward, obs):
        if np.random.rand() <= self.epsilon:
            return self.action_space[np.random.randint(0, len(self.action_space))]
        else:
            # action_prob = self.model.predict()
            # return np.argmax(action_prob)
            return []        

    def study(self):
        """Learn models from memory or by replaying.
        """
        if len(self.memory) < self.batch_size:
            return # not study

        def do_with_play(curstate, action, reward, state, game_over):
            target = self.model.predict(curstate)
            print(target)
            # print(curstate, action, reward, state, game_over)
            return None

        self._replay(self.batch_size, do_with_play)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _replay(self, batch_size, do_with_play):
        """

        do_with_play (curstate, action, reward, state, game_over)-> ??
        """
        minibatch = random.sample(self.memory, batch_size)
        [do_with_play(*play_tuple)
         for play_tuple in minibatch]

    def _state_preprocessor(self, state):
        return np.reshape(state, [1, self.state_size[0]])

    def remember(self, action, reward, state, game_over):
        """Remember current information.

        'memory' will store info formatted as:
            (curstate, action, reward, state, game_over)
        """
        # print(state)
        curstate = self._state
        self._state = self._state_preprocessor(state)
        if curstate is None:
            print('Init state is not set! Call set_state() at each episode.')
            raise
        self.memory.append((curstate, action, reward, self._state, game_over))
    
    def load(self, file_name):
        pass

    def save(self, file_name):
        pass

if __name__ == "__main__":
    # predefined parameters
    n_episodes = 2

    game = FlappyBird()
    trainer = Trainer(game, MyAgent)
    trainer.train(n_episodes)
    # trainer.play()


