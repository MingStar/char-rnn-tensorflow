# char-rnn-tensorflow

Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level and word-level language models in Python using Tensorflow.

Originally [written by Sherjil Ozair](https://github.com/sherjilozair/char-rnn-tensorflow), which was inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

## Prerequisites:


- [Tensorflow](http://www.tensorflow.org)
- Other Python libraries:
    - [Gensim](https://radimrehurek.com/gensim/) for optionally using a [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) embedding
    - [PyYAML](http://pyyaml.org/) for storing human readable model info
    - [CherryPy](http://www.cherrypy.org/) for running a simple sampling web service

#### PIP Installation for Python libraries:

```
$ pip install pyyaml cherrypy gensim
```

## Training

Main Command: `$ python train.py`

1. Input data are to be preprocessed, concatenated and saved as one big text file named `input.txt`
in the sub folder of `data/` folder (e.g. see `data/tinyshakespeare`)
2. run `$ python train.py` with argument `--data_dir` pointing to the above data sub folder
3. The model will be saved in the folder specified by `--save_dir`. The best model, in terms of
minimum training loss so far, will be saved in the `best/`
subfolder of the save folder.

Detail command line arguments for running `python train.py`:

```
$ python train.py -h
usage: train.py [-h] [--data_dir DATA_DIR] [--save_dir SAVE_DIR]
                [--rnn_size RNN_SIZE] [--num_layers NUM_LAYERS]
                [--model MODEL] [--batch_size BATCH_SIZE]
                [--seq_length SEQ_LENGTH] [--num_epochs NUM_EPOCHS]
                [--save_every SAVE_EVERY] [--grad_clip GRAD_CLIP]
                [--learning_rate LEARNING_RATE] [--decay_rate DECAY_RATE]
                [--init_from INIT_FROM]
                [--word2vec_embedding WORD2VEC_EMBEDDING] [--dropout DROPOUT]
                [--print_every PRINT_EVERY] [--word_level]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   data directory containing input.txt (default:
                        data/tinyshakespeare)
  --save_dir SAVE_DIR   directory to store checkpointed models (default: save)
  --rnn_size RNN_SIZE   size of RNN hidden state (default: 128)
  --num_layers NUM_LAYERS
                        number of layers in the RNN (default: 3)
  --model MODEL         rnn, gru, or lstm (default: lstm)
  --batch_size BATCH_SIZE
                        minibatch size (default: 50)
  --seq_length SEQ_LENGTH
                        RNN sequence length (default: 50)
  --num_epochs NUM_EPOCHS
                        number of epochs (default: 50)
  --save_every SAVE_EVERY
                        save frequency (default: 1000)
  --grad_clip GRAD_CLIP
                        clip gradients at this value (default: 5.0)
  --learning_rate LEARNING_RATE
                        learning rate (default: 0.002)
  --decay_rate DECAY_RATE
                        decay rate for rmsprop (default: 0.97)
  --init_from INIT_FROM
                        continue training from saved model at this path. Path
                        must contain files saved by previous training process:
                        'config.pkl' : configuration; 'chars_vocab.pkl' :
                        vocabulary definitions; 'checkpoint' : paths to model
                        file(s) (created by tf). Note: this file contains
                        absolute paths, be careful when moving files around;
                        'model.ckpt-*' : file(s) with model definition
                        (created by tf) (default: None)
  --word2vec_embedding WORD2VEC_EMBEDDING
                        filename for the pre-train gensim word2vec model
                        (default: None)
  --dropout DROPOUT     probability of dropouts for each cell's output
                        (default: 0)
  --print_every PRINT_EVERY
                        print stats of training every n steps (default: 10)
  --word_level          if specified, split text by space on word level,
                        otherwise, spilt text on character level (default:
                        False)
```



## Sampling

There are 2 sampling methods:
* one-off sampling from command line
* multiple sampling as a web service

### Sampling using command line:

Main Command: `$ python sample.py`

Detail command line arguments:
```
$ python sample.py -h
usage: sample.py [-h] [--save_dir SAVE_DIR] [-n N] [--prime PRIME]
                 [--sample SAMPLE] [--temperature TEMPERATURE] [--word_level]

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   model directory to store checkpointed models (default:
                        save)
  -n N                  number of characters to sample (default: 500)
  --prime PRIME         prime text (default: The)
  --sample SAMPLE       0 to use argmax at each timestep, 1 to sample at each
                        timestep, 2 to sample on spaces (default: 1)
  --temperature TEMPERATURE
                        temperature for sampling, within the range of (0,1]
                        (default: 1.0)
  --word_level          if specified, split text by space on word level,
                        otherwise, spilt text on character level (default:
                        False)
```


### Sampling using the CherryPy web service

1. To run the web service: `python sample_server.py`
2. Visit [http://127.0.0.1:8080?prime=The&n=200&sample_mode=2](http://127.0.0.1:8080?prime=The&n=200&sample_mode=2) in the browser.

Detail command line arguments to run the service:
```
$ python sample_server.py -h
usage: sample_server.py [-h] [--port PORT] [--production]
                        [--save_dir SAVE_DIR] [--word_level]

optional arguments:
  -h, --help           show this help message and exit
  --port PORT          port the server runs on (default: 8080)
  --production         specify whether the server runs in production
                       environment or not (default: False)
  --save_dir SAVE_DIR  directory to restore checkpointed models (default:
                       save)
  --word_level         if specified, split text by space on word level,
                       otherwise, spilt text on character level (default:
                       False)
```

#### Web service API parameters:

* `prime`: initial text to prime the network
* `n`: number of tokens to sample
* `sample_mode`:
    * `0` to use argmax at each timestep
    * `1` to sample at each timestep
    * `2` to sample on spaces
    
    
NB: for both command-line and web-server sampling methods, pointing argument `SAVE_DIR` to
the value of `SAVE_DIR` in the training step will use the **latest** model trained so far, to use the **best** model, point
`SAVE_DIR` to `SAVE_DIR` + `'/best/'` from the training step.    


## Improvements made in this repo:

1. Allow word-level tokens, separated by spaces (enable by using the argument flag `--word-level` when running train.py)
1. Save the best model (in terms of minimum training loss) so far in the 'best' subfolder
1. Options to use gensim word2vec embedding
1. Add a web service for sampling (with CherryPy, see sample_sever.py)
1. Temperature [Pull request #28](https://github.com/sherjilozair/char-rnn-tensorflow/pull/28)
1. Dropouts [Pull request #35](https://github.com/sherjilozair/char-rnn-tensorflow/pull/35)



#### License

The MIT License

#### Contact

For questions and usage issues, please contact mingstar215@gmail.com

