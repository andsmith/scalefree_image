import random
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Constant
import numpy as np


class NormalLayer(Layer):
    """
    Just look at the input space with regular RELU units.
    """
    def __init__(self, output_dim,sharpness, grad_sharpness, **kwargs):
        self.output_dim = output_dim
        self.initializer = RandomUniform(-1.0, 1.0)
        self.kernel = None
        super(NormalLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, input_shape[1]),
                                      initializer=self.initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                   shape=(self.output_dim,),
                                   initializer='zeros',
                                   trainable=True)
        super(NormalLayer, self).build(input_shape)

    def call(self, x):
        excitation = K.dot(x, K.transpose(self.kernel)) + self.bias
        p = K.tanh(excitation)
        return p
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'initializer': self.initializer
        }
        base_config = super(NormalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))