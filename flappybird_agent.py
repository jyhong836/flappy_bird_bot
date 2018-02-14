import numpy as np
from ple.games.flappybird import FlappyBird

from fb_trainer import Trainer

class AgentType(object):
    def __init__(self, action_space):
        pass

    def act(self, reward, obs):
        pass

class MyAgent(AgentType):
    """My agent

        A simple agent playing FlappyBird.
    """

    def __init__(self, action_space, batch_size=100):
        self.action_space = action_space

        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.batch_size = batch_size

        self.memory = []

        # private
        self._state = []

        # init model
        self.model = self._build_model()
    
    def _build_model(self):
        model = [] # TODO build model using Keras
        return model

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

    def _state_preprocessor(self, state):
        return state

    def remember(self, action, reward, state, game_over):
        """Remember current information.

        'memory' will store info formatted as:
            (curstate, action, reward, state, game_over)
        """
        print(state)
        curstate = self._state
        self._state = self._state_preprocessor(state)

        self.memory.append((curstate, action, reward, state, game_over))
    
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


