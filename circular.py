import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
import numpy as np
import logging


class InitRadiiRandom(Initializer):
    """ Initializer for initialization of standard deviation
    """

    def __init__(self, num):
        self._num = num
        pass

    def __call__(self, shape, dtype=None, **kwargs):
        spread = [1.5, 1.5]
        sigmas = (np.random.rand(self._num) ** 1.0 - spread[0]) * (spread[1] - spread[0])
        sigmas = np.clip(sigmas, spread[0], spread[1])
        log_sigmas = np.log(sigmas)
        return log_sigmas


class CircleLayer(Layer):
    """
    Layer of circular units, trainable paramters are center (x,y) and radius r

    """

    def __init__(self, output_dim, sharpness=1000.0, grad_sharpness=3.0, initializer=None, **kwargs):
        self.output_dim = output_dim
        self._sharpness = sharpness
        self.grad_sharpness = grad_sharpness

        if not initializer:
            self.initializer = RandomUniform(-1.0, 1.0)
        else:
            self.initializer = initializer
        self.centers = None
        self.radii = None
        self.sharpness = None
        super(CircleLayer, self).__init__(**kwargs)
        logging.info(f"CircleLayer initialized with grad_sharpness={self.grad_sharpness}, sharpness={self._sharpness}")

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.radii = self.add_weight(name='radii',
                                     shape=(self.output_dim,),
                                     initializer=InitRadiiRandom(self.output_dim),
                                     trainable=True)
        self.sharpness = self.add_weight(name='sharpness',
                                         shape=(self.output_dim,),
                                         initializer=Constant(self._sharpness),
                                         trainable=False)
        super(CircleLayer, self).build(input_shape)

    def call(self, x):
        radii = K.exp(self.radii)
        C = K.expand_dims(self.centers)
        sqdist = K.transpose(C-K.transpose(x))**2.0
        dist = K.sqrt(tf.reduce_sum(sqdist, 1))
        excitation = (radii - dist) / radii

        @tf.custom_gradient
        def sharp_with_false_grad(excitation_arg):
            # Forward pass: sharp activation
            forward_result = K.tanh(self.sharpness * excitation_arg)

            def grad_fn(dy):
                # Backward pass: gradient of tanh(cos_theta) instead of tanh(sharpness * cos_theta)
                false_grad = (1.0 - K.tanh(excitation_arg*self.grad_sharpness)**2) * \
                    self.grad_sharpness  # derivative of tanh(cos_theta)
                return dy * false_grad

            return forward_result, grad_fn

        activation = sharp_with_false_grad(excitation)

        return activation

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(CircleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
