import os
import threading
import cherrypy
import argparse

from six.moves import cPickle
from model import Model

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default='8080',
                       help='port the server runs on')
    parser.add_argument('--production', action="store_true",
                        help="specify whether the server runs in production environment or not")
    parser.add_argument('--model_dir', type=str, default='save',
                       help='directory to restore checkpointed models')
    args = parser.parse_args()
    server_config = {'server.socket_port': args.port}
    if args.production:
        server_config['environment'] = 'production'
    cherrypy.config.update(server_config)
    cherrypy.quickstart(SampleServer(args.model_dir))


class SampleServer(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.lock = threading.Lock()
        self.threaded_models = {}
        self._load()

    def _load(self):
        with open(os.path.join(self.model_dir, 'chars_vocab.pkl'), 'rb') as f:
            self.chars, self.vocab = cPickle.load(f)
        with open(os.path.join(self.model_dir, 'config.pkl'), 'rb') as f:
            self.saved_args = cPickle.load(f)

    def _get_model_for_current_thread(self):
        tid = threading.get_ident()
        if tid in self.threaded_models:
            return self.threaded_models[tid]
        # else
        self.lock.acquire()
        if not tid in self.threaded_models:
            model = Model(self.saved_args, infer=True)
            session = tf.Session()
            with session.as_default():
                tf.initialize_all_variables().run()
                saver = tf.train.Saver(tf.all_variables())
                ckpt = tf.train.get_checkpoint_state(self.model_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(session, ckpt.model_checkpoint_path)
                    print("Loaded {}".format(ckpt.model_checkpoint_path))
            self.threaded_models[tid] = (model, session)
        self.lock.release()
        return self.threaded_models[tid]

    @cherrypy.expose
    def index(self, prime='The ', n=200, sample_mode=2):
        model, session = self._get_model_for_current_thread()
        with session.as_default():
            return model.sample(session, self.chars, self.vocab, n, prime, sample_mode)


if __name__ == '__main__':
    main()