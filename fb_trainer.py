import numpy as np
from ple import PLE
import os.path, datetime
from fb_logger import train_logger

class Trainer:
    def __init__(self, game, agent,
                 max_episode_time = 100000,
                 display_screen   = True,
                 save_folder      = None,
                 save_name        = 'fbtr',
                 save_freq        = 0,
                 n_episodes       = 100,
                 max_noops        = 2,
                 fps              = 30,
                 frame_skip       = 3):
        num_steps = 1
        # force_fps = False  # slower speed
        # display_screen = True  # Set to false if there is no screen available

        # init parameters
        # define how many actions will be ignored at the beginning.
        self.max_noops        = max_noops
        self.n_episodes       = n_episodes
        self.cum_n_episodes   = 0
        self.max_episode_time = max_episode_time
        self.save_folder      = save_folder
        self.save_name        = save_name
        self.save_freq        = save_freq # freq of auto-save network

        # make a PLE instance.
        self.ple = PLE(
            game, 
            fps=fps, 
            frame_skip=frame_skip, 
            num_steps=num_steps,
            force_fps=True, # 'False' only for better visual
            display_screen=display_screen,
            state_preprocessor=lambda state: np.array(list(state.values())))

        self.agent = agent
        self.agent.setup(self.ple.getGameStateDims(),self.ple.getActionSet())

        # self.getState = lambda: self.ple.getScreenRGB()
        self.getState = lambda: self.ple.getGameState()

        # create logger
        self.logger = train_logger(save_folder)

    def _get_time(self):
        return str(datetime.datetime.now())

    def load(self, filename):
        if not self.save_folder is None and not filename is None:
            # TODO load cum_n_episodes
            fn = os.path.join(self.save_folder, filename)
            if os.path.isfile(fn):
                print('Load: ' + fn)
                self.agent.load(fn)
            else:
                print('Not found file: '+fn)

    def save(self):
        if not self.save_folder is None and not self.save_name is None:
            # TODO save cum_n_episodes
            filename = os.path.join(
                self.save_folder, self.save_name+'.h5')
            # self.save_folder, 'fbtr_'+self._get_time()+'.h5')
            print('save: '+filename)
            self.agent.save(filename)
    
    def save_screen(self):
        if not self.save_folder is None:
            filename = os.path.join(
                self.save_folder, 'screen_'+self._get_time()+'.png')
            print('try to save screen to '+filename)
            self.ple.saveScreen(filename)

    def _random_action(self):
        """Do a random number of NOOP's

        This is useful when first several actions are nonsense and they can be used to init rewards or states.
        """
        # for _ in range(np.random.randint(0, self.max_noops)):
        for _ in range(self.max_noops):
            self.ple.act(self.ple.NOOP)
        return self.ple.act(self.ple.NOOP)

    def train(self, n_episodes=None):
        """Train model with 'n_episodes' game plays.
        """
        if n_episodes is None:
            n_episodes = self.n_episodes
        self.cum_n_episodes += n_episodes

        def on_step(action, reward, state, game_over):
            self.agent.remember(action, reward, state, game_over)

        def on_game_over(total_reward, ep, n_episodes):
            loss, epsilon = self.agent.study()
            print('[epi: {}/{}], r: {}, loss: {}, epsi: {}'.\
                format(ep+1, n_episodes, total_reward, loss, epsilon))
            self.logger.log(ep+1, total_reward, loss)

        for ep in range(n_episodes):
            self._run(on_step=on_step,
                      on_game_over=lambda total_reward: on_game_over(total_reward, ep, n_episodes))
                # on_init=lambda state: self.agent.set_state(state), 
            if self.save_freq > 0 and ep % self.save_freq == 0:
                self.save()

    def play(self):
        """Play using the trained agent
        """
        self._run(on_game_over=lambda total_reward: print(
            '[Game over] score: {}'.format(total_reward)),
            on_step=lambda action, reward, state, game_over: print('action: {}, reward: {}, gg: {}'.format(action, reward, game_over)),
            do_explore=False)

    def _run(self,
        on_step=lambda action, reward, state, game_over: None,
        on_game_over=lambda total_reward: None,
        do_explore=True):
        """Run the game with agent

        Parameters:
            on_step(action, reward, state, game_over)
            on_game_over(time)
        """
        if self.cum_n_episodes == 0:
            print('[WARN] agent is not trained yet.')
        self.ple.reset_game()
        reward = self._random_action()
        state = self.getState()
        total_reward = 0.0
        # on_init(state)

        for time in range(self.max_episode_time):
            # print(f'state: {state}')
            action = self.agent.act(state)
            reward, state, game_over = self.ple.act(action), self.getState(), self.ple.game_over()
            total_reward += reward
            # if game_over:
            #     print(f'game over reward: {reward}')
            # reward = reward if not game_over else -10

            on_step(action, reward, state, game_over)

            if game_over:
                on_game_over(total_reward)
                break
        else:
            print('Game is not ended when timeout{}. Try increase "max_episode_time" to solve.'.format(
                self.max_episode_time))

