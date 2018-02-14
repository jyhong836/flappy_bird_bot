import numpy as np
from ple import PLE

class Trainer:
    def __init__(self, game, agentType, max_episode_time=10000):
        fps = 30  # fps we want to run at

        frame_skip = 2
        num_steps = 1
        force_fps = False  # slower speed
        display_screen = True  # Set to false if there is no screen available

        # init parameters
        # define how many actions will be ignored at the beginning.
        self.max_noops = 2
        self.cum_n_episodes = 0
        self.max_episode_time = max_episode_time

        # make a PLE instance.
        self.ple = PLE(
            game, 
            fps=fps, 
            frame_skip=frame_skip, 
            num_steps=num_steps,
            force_fps=force_fps, 
            display_screen=display_screen,
            state_preprocessor=lambda state: np.array(list(state.values())))

        self.agent = agentType(
            self.ple.getGameStateDims(),
            self.ple.getActionSet(),
            batch_size=self.max_noops+20)

        # self.getState = lambda: self.ple.getScreenRGB()
        self.getState = lambda: self.ple.getGameState()

    def load(self, filename):
        self.agent.load(filename)  # TODO load cum_n_episodes

    def save(self, filename):
        self.agent.save(filename)  # TODO save cum_n_episodes

    def _random_action(self):
        """Do a random number of NOOP's

        This is useful when first several actions are nonsense and they can be used to init rewards or states.
        """
        # for _ in range(np.random.randint(0, self.max_noops)):
        for _ in range(self.max_noops):
            self.ple.act(self.ple.NOOP)
        return self.ple.act(self.ple.NOOP)

    def train(self, n_episodes):
        """Train model with 'n_episodes' game plays.
        """
        self.cum_n_episodes += n_episodes

        def on_step(action, reward, state, game_over):
            self.agent.remember(action, reward, state, game_over)

        def on_game_over(time, ep, n_episodes):
            print(
                f'[episode: {ep+1}/{n_episodes}], score: {time}, epsilon: {self.agent.epsilon}')
            self.agent.study()

        for ep in range(n_episodes):
            self._run(on_step=on_step,
                      on_game_over=lambda time: on_game_over(time, ep, n_episodes))

    def play(self):
        """Play using the trained agent
        """
        self._run(on_game_over=lambda time: print(
            f'[Game over] score: {time}'))

    def _run(self,
             on_step=lambda action, reward, state, game_over: None,
             on_game_over=lambda time: None):
        """Run the game with agent

        Parameters:
            on_step(action, reward, state, game_over)
            on_game_over(time)
        """
        if self.cum_n_episodes == 0:
            print('[WARN] agent is not trained yet.')
        self.ple.reset_game()
        reward = self._random_action()
        for time in range(self.max_episode_time):
            state = self.getState()
            action = self.agent.act(reward, state)
            reward = self.ple.act(action)
            game_over = self.ple.game_over()

            on_step(action, reward, state, game_over)

            if game_over:
                on_game_over(time)
                break
        else:
            print(f'Game is not ended when timeout{self.max_episode_time}.')
