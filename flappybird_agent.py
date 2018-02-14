import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird

# predefined parameters
n_episodes = 100
nb_frames = 15000

max_episode_time = 500

class MyAgent():
    """My agent

        A simple agent playing FlappyBird.
    """

    def __init__(self, action_space, batch_size=100):
        self.action_space = action_space

        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.batch_size = batch_size

        self.memory = []

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
        

    def study_replay(self):
        """Learn models from memory or by replaying.
        """
        if len(self.memory) < self.batch_size:
            return # not study

    def remember(self):
        """Remember (state, action, ...)
        """
        pass
    
    def load(self, file_name):
        pass

    def save(self, file_name):
        pass

class Trainer:
    def __init__(self, game):
        fps = 30  # fps we want to run at

        frame_skip = 2
        num_steps = 1
        force_fps = False  # slower speed
        display_screen = True  # Set to false if there is no screen available

        # init parameters
        self.reward = 0.0
        self.max_noops = 20
        self.cum_n_episodes = 0

        # make a PLE instance.
        p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
                force_fps=force_fps, display_screen=display_screen)

        self.ple = p
        # self.agent = agent
        self.agent = MyAgent(self.ple.getActionSet())

        self.getState = lambda: self.ple.getScreenRGB()

    # def _init_training(self):
    #     """Init the trainer
    #     """
    #     self._random_action()

    def load(self):
        pass
    def save(self):
        pass
  
    def _random_action(self):
        """Do a random number of NOOP's

        This is useful when first several actions are nonsense and they can be used to init rewards or states.
        """
        for _ in range(np.random.randint(0, self.max_noops-1)):
            self.ple.act(self.ple.NOOP)
        return self.ple.act(self.ple.NOOP)

    def train(self, n_episodes):
        self.ple.reset_game()
        reward = self._random_action() # get initial reward with random actions.
        self.cum_n_episodes += n_episodes
        for ep in range(n_episodes):
            self.ple.reset_game()
            state = self.getState()
            # Run game
            for time in range(max_episode_time):
                action = self.agent.act(reward, state)
                reward = self.ple.act(action)
                game_over = self.ple.game_over()
                self.agent.remember()
                if game_over:
                    # self.agent.study()
                    print(
                        f'[episode: {ep}/{n_episodes}], score: {time}, epsilon: {self.agent.epsilon}')
                    break
            else:
                print(f'Game is not ended when timeout{max_episode_time}.')
            self.agent.study_replay()

    def play(self):
        """Play using the trained agent
        """
        if self.cum_n_episodes == 0:
            print('[WARN] agent is not trained yet.')
        self.ple.reset_game()
        reward = self._random_action()
        state = self.getState()
        for time in range(max_episode_time):
            action = self.agent.act(reward, state)
            reward = self.ple.act(action)
            game_over = self.ple.game_over()
            if game_over:
                print(
                    f'[Game over] score: {time}, epsilon: {self.agent.epsilon}')
                break
        else:
            print(f'Game is not ended when timeout{max_episode_time}.')


game = FlappyBird()
trainer = Trainer(game)
trainer.train(n_episodes)
# trainer.play()


