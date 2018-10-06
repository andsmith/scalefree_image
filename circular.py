import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np


class InitRadiiRandom(Initializer):
    """ Initializer for initialization of standard deviation
    """

    def __init__(self, num):
        self._num = num
        pass

    def __call__(self, dtype=None):
        spread = [.05, 1.0]
        sigmas = (np.random.rand(self._num) - spread[0]) * (spread[1] - spread[0])
        return sigmas


class CircleLayer(Layer):
    """
    Layer of circular units, trainable paramters are center (x,y) and radius r

    """
    def __init__(self, output_dim, sharpness=1000.0, initializer=None, **kwargs):
        self.output_dim = output_dim
        self._sharpness = sharpness

        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        self.centers = None
        self.radii = None
        self.sharpness = None
        super(CircleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers', 
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.radii = self.add_weight(name='radii',
                                     shape=(self.output_dim,),
                                     initializer=InitRadiiRandom(self.output_dim),
                                     trainable=True)
        self.sharpness = self.add_weight(name = 'sharpness',
                                         shape = (self.output_dim,),
                                         initializer=Constant(self._sharpness),
                                         trainable=False)
        super(CircleLayer, self).build(input_shape)

    def call(self, x):
        sharp = self.sharpness
        radii = K.log(1.0 + K.exp(self.radii))  # all radii must be positive, so use the soft-ReLu transform
        C = K.expand_dims(self.centers)
        sqdist = K.transpose(C-K.transpose(x))**2.0
        dist = K.sqrt(tf.reduce_sum(sqdist, 1))
        r = (radii - dist)  * sharp
        p = K.tanh(r)
        return p

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(CircleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
