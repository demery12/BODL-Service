#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request, Markup
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type
import tensorflow as tf

#initalize our flask app
app = Flask(__name__)

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


@app.route('/')
def index():
   return render_template("index.html")

@app.route('/about/')
def about():
   return render_template("about.html")

@app.route('/predict/',methods=['GET','POST'])
def predict(model_loaded=False):
  
   drink_up = sample()
   drink_up = drink_up.replace('\n','<br>')
   drink_up = drink_up.replace('\t','&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;')

   return drink_up

if __name__ == "__main__":
   #decide what port to run the app in
   port = int(os.environ.get('PORT', 5000))
   #run the app locally on the givn port
   app.run(host='0.0.0.0', port=port)
