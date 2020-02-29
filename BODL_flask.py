from flask import Flask, render_template, request, Markup
import numpy as np
from six.moves import cPickle

from model import Model

from six import text_type
import tensorflow as tf
import os

# initalize our flask app
BODL_app = Flask(__name__)


def convert_pkl_for_windows():
    a = open(os.path.join('good_save', 'config.pkl'), "rb").readlines()  # read pickle file line by line
    a = map(lambda x: x.replace(b"\r\n", b"\n"), a)  # replace \r\n with \n
    with open(os.path.join('good_save', 'config.pkl'), "wb") as j:  # write back to file in binary mode
        for i in a:
            j.write(i)


def sample():
    with open(os.path.join('good_save', 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join('good_save', 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        print(sess)
        tf.global_variables_initializer().run()
        print(tf.global_variables())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('good_save')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return (model.sample(sess, chars, vocab, 500, u' ',
                                 1).encode('utf-8'))


@BODL_app.route('/')
def index():
    return render_template("index.html")


@BODL_app.route('/about/')
def about():
    return render_template("about.html")


@BODL_app.route('/predict/', methods=['GET', 'POST'])
def predict(model_loaded=False):
    drink_up = sample()
    print(drink_up)
    drink_up = drink_up.replace(b'\n', b'<br>')
    drink_up = drink_up.replace(b'\t', b'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;')

    return drink_up


if __name__ == "__main__":
    # decide what port to run the app in
    # port = int(os.environ.get('PORT', 5000))
    # run the app locally on the given port
    BODL_app.run(host='127.0.0.1')
