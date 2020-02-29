from six.moves import cPickle

from model import Model

from six import text_type
import tensorflow as tf
import os

def sample():
    with open(os.path.join('good_save', 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join('good_save', 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('good_save')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return (model.sample(sess, chars, vocab, 500000, u' ',
                                 1).encode('utf-8'))


def sample_and_write(n):
    with open("static_samples.txt", 'a') as f:
        out = sample()
        f.writelines(str(out))


sample_and_write(10)
