"""A logger for tensorflow board
"""

import tensorflow as tf
from numpy.random import rand

class train_logger(object):
    """ Creates an "empty model" that writes Tensorflow summaries. Can
        visualize these summaries with Tensorboard.
    """

    def _create_train_variables(self):
        with tf.name_scope("train"):
            # create variables
            reward = tf.Variable(0.0, name="reward", trainable=False)
            loss = tf.Variable(0.0, name="loss", trainable=False)
            # create summaries
            reward_sum = tf.summary.scalar("reward", reward)
            loss_sum = tf.summary.scalar("loss", loss)
            # tf.summary.scalar('epsilon', )
        with tf.name_scope("test"):
            # create variables
            te_reward = tf.Variable(0.0, name="reward", trainable=False)
            # create summaries
            ter_sum = tf.summary.scalar("reward", te_reward)
        # merge op
        summary_op = tf.summary.merge([reward_sum, loss_sum])
        te_summary_op = tf.summary.merge([ter_sum])
        return reward, loss, te_reward, summary_op, te_summary_op


    def __init__(self, summary_dir):
        self.summary_dir = summary_dir

        # a simple session
        sess = tf.Session()
        self.reward, self.loss, self.te_reward, summary_op, te_summary_op = self._create_train_variables()
        # create writer
        summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

        # initialize
        tf.global_variables_initializer().run(session=sess)

        self.sess = sess
        self.summary_op = summary_op
        self.te_summary_op = te_summary_op
        self.summary_writer = summary_writer

    def log(self, step, reward, loss, is_test=False):
        """Log data by one step.
        """
        if is_test:
            feed_dict = {
                self.te_reward: reward
            }
            self.summary_writer.add_summary(self.sess.run([self.te_summary_op], feed_dict)[0], step)
        else:
            feed_dict = {
                self.reward: reward,
                self.loss: loss,
            }
            self.summary_writer.add_summary(self.sess.run(
                [self.summary_op], feed_dict)[0], step)
    
    def close(self): # FIXME not closed
        self.summary_writer.close()

if __name__ == "__main__": # for test
    logger = train_logger('save')  # logdir='save'

    fake_info = rand(100, 2) # random data

    for step, infoset in enumerate(fake_info):
        if step % 10 == 0:
            logger.log(step, infoset[0], infoset[1], is_test=True)
        logger.log(step, infoset[0], infoset[1], is_test=False)

    logger.close()

