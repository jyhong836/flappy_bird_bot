import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird


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
    def __init__(self, game, max_episode_time=10000):
        fps = 30  # fps we want to run at

        frame_skip = 2
        num_steps = 1
        force_fps = False  # slower speed
        display_screen = True  # Set to false if there is no screen available

        # init parameters
        self.max_noops = 20 # define how many actions will be ignored at the beginning.
        self.cum_n_episodes = 0
        self.max_episode_time = max_episode_time

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

    def load(self, filename):
        self.agent.load(filename)

    def save(self, filename):
        self.agent.save(filename)
  
    def _random_action(self):
        """Do a random number of NOOP's

        This is useful when first several actions are nonsense and they can be used to init rewards or states.
        """
        for _ in range(np.random.randint(0, self.max_noops-1)):
            self.ple.act(self.ple.NOOP)
        return self.ple.act(self.ple.NOOP)

    def train(self, n_episodes):
        self.cum_n_episodes += n_episodes

        def on_step():
            self.agent.remember()

        def on_game_over(time, ep, n_episodes):
            print(f'[episode: {ep}/{n_episodes}], score: {time}, epsilon: {self.agent.epsilon}')
            self.agent.study_replay()

        for ep in range(n_episodes):
            self._play(on_step=on_step,
                      on_game_over=lambda time:on_game_over(time, ep, n_episodes))

    def play(self):
        self._play(on_game_over=lambda time: print(
            f'[Game over] score: {time}'))

    def _play(self, on_step=lambda: None, on_game_over=lambda:None):
        """Play using the trained agent
        """
        if self.cum_n_episodes == 0:
            print('[WARN] agent is not trained yet.')
        self.ple.reset_game()
        reward = self._random_action()
        state = self.getState()
        for time in range(self.max_episode_time):
            action = self.agent.act(reward, state)
            reward = self.ple.act(action)
            game_over = self.ple.game_over()
            on_step()
            if game_over:
                on_game_over(time)
                break
        else:
            print(f'Game is not ended when timeout{self.max_episode_time}.')


if __name__ == "__main__":
    # predefined parameters
    n_episodes = 2

    game = FlappyBird()
    trainer = Trainer(game, max_episode_time = 500)
    trainer.train(n_episodes)
    # trainer.play()


