from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend

class ModelFactory():
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
        model.add(Dense(32, input_shape=state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=learning_rate))
        return model

    @staticmethod
    def naive_dqn_v1(state_size, action_size, learning_rate):
        """ Neural Net for Deep-Q learning Model
        """
        print('Build Deep Q-learnig Model.')

        def huber_loss(target, prediction):
            """sqrt(1+error^2)-1"""
            return keras.backend.mean(keras.backend.sqrt(1+keras.backend.square(prediction - target))-1, axis=-1)

        model = Sequential()
        # print('### '+repr(state_size))
        model.add(Dense(32, input_shape=state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=learning_rate))
        return model
