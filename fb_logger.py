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
            tf.summary.scalar("reward", reward)
            tf.summary.scalar("loss", loss)
        return reward, loss


    def __init__(self, summary_dir):
        self.summary_dir = summary_dir

        # a simple session
        sess = tf.Session()
        self.loss, self.reward = self._create_train_variables()
        # merge op
        summary_op = tf.summary.merge_all() 
        # create writer
        summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

        # initialize
        tf.global_variables_initializer().run(session=sess)

        self.sess = sess
        self.summary_op = summary_op
        self.summary_writer = summary_writer

    def log(self, step, reward, loss):
        """Log data by one step.
        """
        feed_dict = {
            self.reward: reward,
            self.loss: loss,
        }

        # sess.run returns a list, so we have to explicitly
        # extract the first item using sess.run(...)[0]
        summaries = self.sess.run([self.summary_op], feed_dict)[0]
        self.summary_writer.add_summary(summaries, step)
    
    def close(self): # FIXME not closed
        self.summary_writer.close()

if __name__ == "__main__": # for test
    logger = train_logger('save')  # logdir='save'

    fake_info = rand(100, 2) # random data

    for step, infoset in enumerate(fake_info):
        logger.log(step, infoset[0], infoset[1])

    logger.close()

