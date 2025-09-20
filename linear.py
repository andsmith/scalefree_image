import random
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Constant
import numpy as np


class LineLayer (Layer):
    """
    Layer of sharp line units
    Like tanh units, but compress horizontally to a constant width
    """

    def __init__(self, output_dim, sharpness=500.0, initializer=None, **kwargs):
        self.output_dim = output_dim
        self._sharpness = sharpness
        if not initializer:
            self.initializer = RandomUniform(-1.0, 1.0)
        else:
            self.initializer = initializer
        self.centers = None
        self.angles = None
        self.sharpness = None
        super(LineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)

        self.angles = self.add_weight(name='betas',
                                      shape=(self.output_dim,),
                                      initializer=RandomUniform(-np.pi, np.pi),
                                      trainable=True)

        self.sharpness = self.add_weight(name = 'sharpness',
                                         shape = (self.output_dim,),
                                         initializer=Constant(self._sharpness),
                                         trainable=False)
        super(LineLayer, self).build(input_shape)

    def call(self, x):

        unit_vectors = K.concatenate((K.expand_dims(K.cos(self.angles)),
                                      K.expand_dims(K.sin(self.angles))), 1)
        C = K.expand_dims(self.centers)
        vecs = K.transpose(C-K.transpose(x))

        # normalize to unit vectors
        vecs = vecs / (K.sqrt(tf.reduce_sum(vecs**2, axis=1, keepdims=True)) + 1e-10)
        # take the dot product of input vector with each unit vector
        cos_theta = tf.reduce_sum(tf.multiply(vecs, K.transpose(unit_vectors)), axis=1)
        p = K.tanh(self.sharpness * cos_theta) # -0.05 * K.exp(K.log(1.0 + dist**2.0))
        return p

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(LineLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
