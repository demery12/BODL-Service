from flask import Flask, render_template,request, Markup
import numpy as np
from six.moves import cPickle

from model import Model

from six import text_type
import tensorflow as tf
import os

#initalize our flask app
BODL_app = Flask(__name__)

with open(os.path.join('good_save', 'config.pkl'), 'rb') as f:
   saved_args = cPickle.load(f)
with open(os.path.join('good_save', 'chars_vocab.pkl'), 'rb') as f:
   chars, vocab = cPickle.load(f)
model = Model(saved_args, training=False)

"""
def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)
"""

def sample():
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('good_save')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return(model.sample(sess, chars, vocab, 500, u' ',
                               1).encode('utf-8'))


@BODL_app.route('/')
def index():
   return render_template("index.html")

@BODL_app.route('/about/')
def about():
   return render_template("about.html")

@BODL_app.route('/predict/',methods=['GET','POST'])
def predict(model_loaded=False):
  
   drink_up = sample()
   drink_up = drink_up.replace('\n','<br>')
   drink_up = drink_up.replace('\t','&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;')

   return drink_up

if __name__ == "__main__":
   #decide what port to run the app in
   #port = int(os.environ.get('PORT', 5000))
   #run the app locally on the givn port
   BODL_app.run(host='0.0.0.0')
