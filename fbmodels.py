import keras.backend

# print("------- backend: ", keras.backend.backend())
# if "tensorflow" == keras.backend.backend():
#     from tensorflow.python.client import device_lib
#     print("--- type: ", device_lib.list_local_devices())
#     for dv in device_lib.list_local_devices():
#         dt = dv.device_type
#         if "GPU" == dt:
#             print("GPU setting.....")
#             import tensorflow as tf
#             from keras.backend.tensorflow_backend import set_session
#             config = tf.ConfigProto()
#             config.gpu_options.allow_growth = True
#             # config.gpu_options.visible_device_list = "0"
#             set_session(tf.Session(config=config))
#             break

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

# import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# sys.exit()

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
        """ Neural Net for Deep-Q learning Model (https://github.com/DongjunLee/dqn-tensorflow/blob/master/model.py)
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

    @staticmethod
    def naive_dqn_v2(state_size, action_size, learning_rate):
        """ Neural Net for Deep-Q learning Model (https://github.com/DongjunLee/dqn-tensorflow/blob/master/model.py)
        """
        print('Build Deep Q-learnig Model.')

        def huber_loss(target, prediction):
            """sqrt(1+error^2)-1"""
            return keras.backend.mean(keras.backend.sqrt(1+keras.backend.square(prediction - target))-1, axis=-1)

        model = Sequential()
        # print('### '+repr(state_size))
        model.add(Dense(16, input_shape=state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=learning_rate))
        return model
