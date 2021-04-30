from include.layers import *
from include.metrics import *
from include.inits import *
from include.Config import Config

import functools
print = functools.partial(print, flush=True)

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.global_step = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        # self.opt_op = self.optimizer.minimize(self.loss)
        self.opt_op = tf.train.GradientDescentOptimizer(self.optimizer).minimize(self.loss, self.global_step)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN_Align(Model):
    def __init__(self, placeholders, input_dim, output_dim, ILL, sparse_inputs=False, featureless=True, **kwargs):
        super(GCN_Align, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.ILL = ILL
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=Config.ae_learning_rate)
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.exponential_decay(Config.ae_learning_rate, self.global_step, 10, 0.96, False)

        self.build()

    def _loss(self):
        self.loss += align_loss(self.outputs, Config.ae_gamma, Config.ae_neg_K)

    def _accuracy(self):
        pass

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            # act=tf.nn.relu,
                                            act=tf.nn.selu,
                                            dropout=False,
                                            featureless=self.featureless,
                                            sparse_inputs=self.sparse_inputs,
                                            transform=False,
                                            init=trunc_normal,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.output_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            transform=False,
                                            logging=self.logging))
