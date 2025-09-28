import random
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Constant
import numpy as np
import logging


class LineLayer (Layer):
    """
    Layer of sharp line units.
    
    Lines are parameterized by angle and offset from origin.
    
    Like tanh units, but compress horizontally to a constant width
    """

    def __init__(self, output_dim, sharpness=500.0, grad_sharpness=3.0, parameterization="2_param", initializer=None, **kwargs):
        self.output_dim = output_dim
        self._sharpness = sharpness
        self.grad_sharpness = grad_sharpness
        self.param = parameterization
        
        if not initializer:
            self.initializer = RandomUniform(-1.0, 1.0)
        else:
            self.initializer = initializer
        if self.param == "3_param":
            logging.info("#_#_#_#_#_#_#_#_#_#_#_#_#_# Initializing 3-parameterization of lines:  ANGLES + CENTERS")
            self.centers = None
        elif self.param == "2_param":
            logging.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Initializing 2-parameterization of lines:  OFFSETS + ANGLES")
            self.offsets = None
        else:
            raise ValueError(f"Unknown parameterization: {self.param}")
        self.angles = None
        self.sharpness = None
        super(LineLayer, self).__init__(**kwargs)

        logging.info(f"LineLayer initialized with grad_sharpness={self.grad_sharpness}, sharpness={self._sharpness}")

    def build(self, input_shape):
        if self.param == "3_param":
            self.centers = self.add_weight(name='centers',
                                        shape=(self.output_dim, input_shape[1]),
                                        initializer=self.initializer,
                                        trainable=True)
        elif self.param == "2_param":
            self.offsets = self.add_weight(name='offsets',
                                           shape=(self.output_dim, ),
                                           initializer=RandomUniform(-1.,1.),
                                           trainable=True)

        self.angles = self.add_weight(name='angles',
                                      shape=(self.output_dim,),
                                      initializer=RandomUniform(-np.pi, np.pi),
                                      trainable=True)

        self.sharpness = self.add_weight(name='sharpness',
                                         shape=(self.output_dim,),
                                         initializer=Constant(self._sharpness),
                                         trainable=False)
        super(LineLayer, self).build(input_shape)
    

    def call(self, x):
        
        if self.param == "3_param":
            # Excitation for theta, center parameterization:
            unit_vectors = K.concatenate((K.expand_dims(K.cos(self.angles)),
                                        K.expand_dims(K.sin(self.angles))), 1)
            C = K.expand_dims(self.centers)
            vecs = K.transpose(C-K.transpose(x))

            # normalize to unit vectors
            vecs = vecs / (K.sqrt(tf.reduce_sum(vecs**2, axis=1, keepdims=True)) + 1e-10)
            # take the dot product of input vector with each unit vector
            cos_theta = tf.reduce_sum(tf.multiply(vecs, K.transpose(unit_vectors)), axis=1)
            excitation = cos_theta
        elif self.param == "2_param":
            # Excitation for theta, offset parameterization:
            theta = self.angles
            perp_offsets = self.offsets
            # Signed perpendicular distance from input xy to line defined by angle theta and offset perp_offsets
            # Use broadcasting to compute distances for all batch samples and all units
            distances = K.expand_dims(x[:, 0], 1) * K.cos(theta) + K.expand_dims(x[:, 1], 1) * K.sin(theta) - perp_offsets
            excitation = distances
        # Use sharp activation but false gradient
        
        @tf.custom_gradient
        def sharp_with_false_grad(cos_theta_arg):
            # Forward pass: sharp activation
            forward_result = K.tanh(self.sharpness * cos_theta_arg)

            def grad_fn(dy):
                # Backward pass: gradient of tanh(cos_theta) instead of tanh(sharpness * cos_theta)
                false_grad = (1.0 - K.tanh(cos_theta_arg * self.grad_sharpness)**2) * \
                    self.grad_sharpness  # derivative of tanh(cos_theta * grad_sharpness)
                return dy * false_grad

            return forward_result, grad_fn

        p = sharp_with_false_grad(excitation)

        return p

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(LineLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
