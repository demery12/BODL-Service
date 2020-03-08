BODL-Service
===

This project uses Multi-layer Recurrent Neural Networks (LSTM, RNN) to generate cocktail names and recipes. I used the model from [sherjilozair](https://github.com/sherjilozair/char-rnn-tensorflow). Look there for more on the actual model.

I pulled the data for this from https://www.thecocktaildb.com/.

I originally served the site via Flask + NGINX on a DigitalOcean droplet, and pressing the 'cheers' button would actually sample the model.
That was inefficient and more expensive. I have recently start hosting the app statically in an AWS S3 bucket, where a bunch of runs of the model are stored and one is randomly selected each time 'cheers' is pressed.

You can check it out [here](http://dylan-emery.com/BODL-Service/index.html).
