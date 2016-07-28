# char-rnn-tensorflow

# New changes in this repo:

- Save the best model (i.e. minimum training loss) so far in the 'best' subfolder
- Options for using gensim word2vec embedding
- Add a web server for sampling
- Temperature [Pull request #28](https://github.com/sherjilozair/char-rnn-tensorflow/pull/28)
- Dropouts [Pull request #35](https://github.com/sherjilozair/char-rnn-tensorflow/pull/35)


## Requirements
- [Tensorflow](http://www.tensorflow.org)
- python libraries:
    - [PyYAML](http://pyyaml.org/)
    - [CherryPy](http://www.cherrypy.org/)
    - [Gensim](https://radimrehurek.com/gensim/)

Installation for python libraries:

```
pip install pyyaml cherrypy gensim
```

## Sampling with the web server
```
python sample_server.py --save_dir <your_checkpointed_model_directory>
```

When the server is running, visit http://127.0.0.1:8080?prime=The&n=200&sample_mode=2 in the browser.
 

# Readme from the original/upstream repo:

Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow](http://www.tensorflow.org)

# Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`.

To sample from a checkpointed model, `python sample.py`.
# Roadmap
- Add explanatory comments
- Expose more command-line arguments
- Compare accuracy and performance with char-rnn
