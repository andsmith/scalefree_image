import sys
import time
import tensorflow as tf
import tempfile
import shutil
try:
    import cPickle as cp
except ImportError:
    import pickle as cp
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np
import cv2
import time
import os
import logging
from util import make_input_grid, downscale_image, make_central_weights, fade
import argparse
from threading import Thread, Lock
from circular import CircleLayer
from linear import LineLayer
from normal import NormalLayer
import matplotlib.pyplot as plt
import json
from skimage import measure
import matplotlib.gridspec as gridspec
from synth_image import TestImageMaker
from copy import deepcopy

from loop_timing.loop_profiler import LoopPerfTimer as LPT
DIV_TYPES = {'circular': CircleLayer, 'linear': LineLayer, 'sigmoid': NormalLayer}


class ScaleInvariantImage(object):
    """
    Train a feed-forward neural network to approximate an image:
    learn f(x, y) = (r,g,b) using every pixel as a training example

    """

    def __init__(self, image_raw, n_hidden, n_structure, n_div, state_file=None, batch_size=64, sharpness=1000.0, grad_sharpness=3.0, 
                 learning_rate_initial=1.0, downscale=1.0, center_weight_params=None, line_params=3, **kwargs):
        """
        :param image_raw: a HxWx3 or HxW numpy array containing the target image.  Training will be wrt downsampled versions of this image.
        :param n_hidden: number of hidden units in the middle
        :param n_structure: number of structure units
        :param n_div: dictionary containing the number of input units for each division type
        :param n_div_s: number of input units for sigmoid division
        :param state_file: if not None, a file to load the model state from (overrides other args except learning rate, batch size)
        :param image: if not None, a HxWx3 or HxW numpy array containing the target image
        :param batch_size: training batch size
        :param sharpness: sharpness constant for activation function, e.g. f(x) = tanh(x*sharpness) for linear
        :param grad_sharpness: sharpness constant for gradient of activation function, e.g. f'(x) = sharpness * sech^2(x*sharpness)
        :param learning_rate_initial: initial learning rate for Adadelta optimizer
        :param downscale: downscale factor for training image (1.0 = full size, 0.5 = half size, etc)
        :param center_weight_params: if not None, a dict with keys: 'r_inner' (float), 'r_outer' (float), 'w_max' (float), and 'xy_offset' (tuple of floats)
        :param line_params: 2 or 3, parameterization of line units (2 = angle + offset, 3 = angle + center + offset)
        """
        self.image_raw = image_raw
        self._line_params = line_params
        self.n_hidden = n_hidden
        self.n_div = n_div
        self.n_structure = n_structure
        self.batch_size = batch_size
        self.grad_sharpness = grad_sharpness
        self.image_train = None
        self._downscale = downscale
        self._sample_weights = None
        self.sharpness = sharpness
        self.cycle = 0  # increment for each call to train_more()
        self._center_weight_params = center_weight_params
        self._learning_rate = learning_rate_initial
        self._artists = {'circular':{
                            'center_points':[],
                            'curves': []},
                         'linear': {
                             'center_points': [],
                             'lines': []},
                         'sigmoid': { 
                             'bands': []},
                         'output_image': None}
        self.anneal_temp = 0.0
        self._lims_set = False
        
        self.cur_loss = -1

        if state_file is not None:
            # These params can't change (except for updating weights in self._model), so they override the args.
            # (so they can be None in the args)
            state = ScaleInvariantImage._load_state(state_file)
            weights = state['weights']
            self.cycle, self.image_raw, self.n_hidden, self.n_structure, self.n_div,  self.sharpness, self.grad_sharpness, self._downscale = \
                state['cycle'], state['image_raw'], state['n_hidden'], state['n_structure'], state['n_div'], state['sharpness'], state['grad_sharpness'], state['downscale']
            # Checks
            if image_raw is not None and image_raw.shape != self.image_raw.shape:
                logging.warning("Image shape doesn't match loaded state:  %s vs %s, using CMD-line argument (will save with this one too)" %
                                (image_raw.shape, self.image_raw.shape))
                self.image_raw = image_raw
                
            if self._downscale != downscale:
                logging.info("Warning: Command-line is overriding model's downscale factor:  was %.3f, using %.3f" %
                                (self._downscale, downscale))
                self._downscale = downscale
        else:
            weights = None
            
        # Cache this
        self._last_downscale = None
        self._input, self._output = self._make_train(self._downscale)


        self._model = self._init_model()
        if weights is not None:
            logging.info("Restored model weights from file:  %s  (resuming at cycle %i)" % (state_file, self.cycle))
            self._model.set_weights(weights)
        else:
            logging.info("Initialized new model.")

        self._model.compile(loss='mean_squared_error',
                            optimizer=tf.keras.optimizers.Adadelta(learning_rate=self._learning_rate, use_ema=False, ema_momentum=0.99))  # default 0.001

        logging.info("Model compiled with default learning_rate:  %f" % (self._learning_rate,))

    def _make_train(self, downscale, keep_aspect=True):
        
        if self._last_downscale is not None and self._last_downscale == downscale:
            return self._input, self._output
        self._last_downscale = downscale

        self._downscale_level = downscale
        self.image_train = downscale_image(self.image_raw, self._downscale_level)
        

        in_x, in_y = make_input_grid(self.image_train.shape, keep_aspect=keep_aspect)

        grid_shape = in_x.shape
        input = np.hstack((in_x.reshape(-1, 1), in_y.reshape(-1, 1)))
        r, g, b = cv2.split(self.image_train / 255.0)
        output = np.hstack((r.reshape(-1, 1), g.reshape(-1, 1), b.reshape(-1, 1)))
        
        if self._center_weight_params is not None:
            train_img_size_wh = self.image_train.shape[1], self.image_train.shape[0]
            self._weight_grid = make_central_weights(train_img_size_wh, **self._center_weight_params)
            logging.info("Using center-weighted samples with max weight %.1f and sigma %.3f (image shape: %s)" %
                         (self._center_weight_params['w_max'], self._center_weight_params['r_inner'], train_img_size_wh))
            self._sample_weights = self._weight_grid.reshape(-1)
            self.weight_cross_sections = {'x': self._weight_grid[self._weight_grid.shape[0]//2,:],
                                          'y': self._weight_grid[:,self._weight_grid.shape[1]//2]}
        else:
            logging.info("Not using weighted samples.")
            self._sample_weights = None
        logging.info("Made inputs %s spanning [%.3f, %.3f] and [%.3f, %.3f], %i samples total." %
                     (grid_shape, input[:, 0].min(), input[:, 0].max(), input[:, 1].min(), input[:, 1].max(), input.shape[0]))
        self._input, self._output = input, output
        return input, output
    
    def get_div_params(self):
        """
        Get the parameters of the division units.  For L lines, C circles, and S sigmoids, 
        returns:  dict{
            'circular': {'centers': Lx2 array, 'angles': L array}
            'linear': {'centers': Cx2 array, 'radii': C array}
            'sigmoid': {'weights': Sx2 array, 'biases': S array}}
        """
        params = {}
        for layer in self._model.layers:
            if layer.__class__.__name__ == 'CircleLayer':
                weights = layer.get_weights()
                params['circular'] = {'centers': weights[0], 'radii': np.exp(weights[1])}  # radii are stored as log(r)
            elif layer.__class__.__name__ == 'LineLayer':
                weights = layer.get_weights()
                # three param:
                if self._line_params == 3:
                    params['linear'] = {'centers': weights[0], 'angles': weights[1], 'offsets': weights[2]}
                # two param"
                elif self._line_params == 2:
                    params['linear'] = {'offsets': weights[0], 'angles': weights[1]}
            elif layer.__class__.__name__ == 'NormalLayer':
                weights = layer.get_weights()
                params['sigmoid'] = {'weights': weights[0], 'biases': weights[1]}
        return params

    def unit_coords_to_pixels(self, coords_xy, img_shape, orig_aspect=True):
        """
        Convert unit coordinates (x,y) in [-1,1]x[-1,1] to pixel coordinates in [0,w-1]x[0,h-1]
        :param img_shape: (h,w,c) shape of the image
        :param orig_aspect: assume coords are bounded on the narrower dimension
           to use the training image's aspect ratio, otherwise assume square
        """
        w, h = self.image_train.shape[1], self.image_train.shape[0]

        if orig_aspect:
            # unscale to unit square
            aspect_ratio = w / h
            if aspect_ratio > 1:
                x = coords_xy[:,0]
                y = coords_xy[:,1] * aspect_ratio
            else:
                x = coords_xy[:,0] / aspect_ratio
                y = coords_xy[:,1]
        else:
            x = coords_xy[:,0]
            y = coords_xy[:,1]
            
        # now scale to pixel coords
        px = (x + 1.0) * 0.5 * (img_shape[1]-1)
        py = (y + 1.0) * 0.5 * (img_shape[0]-1)
        return np.hstack((px.reshape(-1, 1), py.reshape(-1, 1))).astype(int)
    
    def radius_to_pixel_radius(self, radius, img_shape, orig_aspect=True):
        """
        Convert a radius in unit coordinates to pixel radius
        :param img_shape: (h,w,c) shape of the image
        :param orig_aspect: assume coords are bounded on the narrower dimension
           to use the training image's aspect ratio, otherwise assume square
        """
        w, h = self.image_train.shape[1], self.image_train.shape[0]

        if orig_aspect:
            # unscale to unit square
            aspect_ratio = w / h
            if aspect_ratio > 1:
                r_px = radius * aspect_ratio * 0.5 * (img_shape[0]-1)
            else:
                r_px = radius / aspect_ratio * 0.5 * (img_shape[1]-1)
        else:
            r_px = radius * 0.5 * min(img_shape[0]-1, img_shape[1]-1)
        return r_px.astype(int)


    def draw_div_units(self, ax, output_image=None, margin=0.1, plot_units=False, draw_flags=None):
        """
        # Draw a representation of the division units on the given axis.
        For line units: 
            Put a hollow dot at the center, draw a line segment through it at the angle
        For circle units:
            Draw a circle at the center, and the circle with the radius
        for normal units:
            if "tanh" activation function, draw a band parallel to the line implied
            by the weights & bias, and thickness proportional to the inverse of the weight norm.
        :param ax: a matplotlib axis to draw on
        :param margin: The x and y limits will be [-1, +1] on the larger dimension,
            (-a-margin, a+margin)) on the smaller, where a is min(aspect, 1/aspect)
        """
        
        aspect_ratio = self.image_train.shape[1] / self.image_train.shape[0]
        if aspect_ratio > 1:
             x_lim = (-1.0, 1.0)
             y_lim = (-1/aspect_ratio, 1/aspect_ratio)
        else:
             x_lim = (-aspect_ratio, aspect_ratio)
             y_lim = (-1.0, 1.0)
             
        # Base image for plotting (reuse artist to avoid accumulation)
        if output_image is not None:
            # if norm_colors:
            #     img_to_show = output_image.copy()
            #     for c in range(3):
            #         c_min, c_max = img_to_show[:, :, c].min(), img_to_show[:, :, c].max()
            #         if c_max > c_min:
            #             img_to_show[:, :, c] = (img_to_show[:, :, c] - c_min) / (c_max - c_min)
            # else:
            #     img_to_show = output_image
            img_to_show = output_image
        else:
            return

        image_extent = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]]
        ax.set_anchor('C')
        if self._artists['output_image'] is None:
            self._artists['output_image'] = ax.imshow(
                img_to_show,
                extent=image_extent,
                origin='upper',
                aspect='auto'
            )
        else:
            self._artists['output_image'].set_data(img_to_show)
            self._artists['output_image'].set_extent(image_extent)
        #ax.invert_yaxis()
        alpha=1.0
        line_width = 2.0
        

        params = self.get_div_params()
        if not plot_units:
            for line in self._artists['linear']['lines']:
                line.set_visible(False)
            for center in self._artists['linear']['center_points']:
                center.set_visible(False)
            for band in self._artists['sigmoid']['bands']:
                band.set_visible(False)
            for center in self._artists['circular']['center_points']:
                center.set_visible(False)
            for curve in self._artists['circular']['curves']:
                curve.set_visible(False)
            return
        else:
            # make everything visible
            for line in self._artists['linear']['lines']:
                line.set_visible(True)
            for center in self._artists['linear']['center_points']:
                center.set_visible(True)
            for band in self._artists['sigmoid']['bands']:
                band.set_visible(True)
            for center in self._artists['circular']['center_points']:
                center.set_visible(True)
            for curve in self._artists['circular']['curves']:
                curve.set_visible(True)
        
        ax.invert_yaxis()
        if 'linear' in params:
            if 'centers' in params['linear']:
                centers = params['linear']['centers'].reshape(-1,2)  # switch to (x,y)
                # FLIP Y for display
                #centers = centers[:,::-1]
                centers[:,1] = -centers[:,1]
                angles = params['linear']['angles']
                # rotate for display
                #angles = np.pi/2.0 - angles  
                if len(self._artists['linear']['center_points']) == 0:
                    self._artists['linear']['center_points'].append( ax.plot(centers[:,0], centers[:,1], 
                                                                        'o', color='red', markersize=4, alpha=alpha)[0])
                else:
                    self._artists['linear']['center_points'][0].set_data(centers[:,0], centers[:,1])

                for i in range(self.n_div['linear']):
                    c = centers[i]
                    t = [-3.0, 3.0]
                    dx = np.cos(angles[i])
                    dy = np.sin(angles[i])
                    p0 = c + t[0] * np.array([dy, dx])
                    p1 = c + t[1] * np.array([dy, dx])
                    if len(self._artists['linear']['lines']) <= i:
                        line, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], '-', color='red', alpha=0.5)
                        self._artists['linear']['lines'].append(line)
                    else:
                        self._artists['linear']['lines'][i].set_data([p0[0], p1[0]], [p0[1], p1[1]])

            elif 'offsets' in params['linear']:
                # 2 parameterization of the line (angle, offset from origin)
                offsets = params['linear']['offsets']
                angles = params['linear']['angles']  
                normals = np.vstack((np.cos(angles), 
                                     -np.sin(angles))).T
                for l_i in range(self.n_div['linear']):
                    n = normals[l_i]
                    d = offsets[l_i]
                    # point on line closest to origin:
                    c = n * d
                    t = [-4.0, 4.0]
                    dx = np.cos(angles[l_i])
                    dy = np.sin(angles[l_i])
                    p0= c + t[0] * np.array([dy, dx])
                    p1= c + t[1] * np.array([dy, dx])
                    
                    points = np.array((p0, p1))
                    
                    
                    if len(self._artists['linear']['lines']) <= l_i:
                        line, = ax.plot(points[:,0], points[:,1], '-', color='red', alpha=alpha, linewidth=line_width)
                        self._artists['linear']['lines'].append(line)
                    else:
                        self._artists['linear']['lines'][l_i].set_data(points[:,0], points[:,1])
        if 'circular' in params:
            
            centers = params['circular']['centers']# switch to (x,y)
            centers[:,1] = -centers[:,1]
            radii = params['circular']['radii']

            if len(self._artists['circular']['center_points']) == 0:
                self._artists['circular']['center_points'].append( ax.plot(centers[:,0], centers[:,1], 
                                                                    'o', color='red', markersize=4, alpha=alpha)[0])
            else:
                self._artists['circular']['center_points'][0].set_data(centers[:,0], centers[:,1])
            
            for i in range(centers.shape[0]):
                c = centers[i]
                r = radii[i]
                if len(self._artists['circular']['curves']) <= i:
                    circle_inner = plt.Circle((c[0], c[1]), r, color='blue', fill=False, alpha=alpha, linewidth=line_width)
                    ax.add_artist(circle_inner)
                    self._artists['circular']['curves'].append(circle_inner)
                else:
                    self._artists['circular']['curves'][i].set_radius(r)
                    self._artists['circular']['curves'][i].set_center((c[0], c[1]))
                    
                    
        if 'sigmoid' in params:
            weights = params['sigmoid']['weights']
            # flip Y for display
            weights[:,1] = -weights[:,1]
            biases = params['sigmoid']['biases']
            for i in range(weights.shape[0]):
                w = weights[i]
                b = biases[i]
                norm = np.sqrt(w[0]**2 + w[1]**2)
                if norm > 0:
                    ax.plot([-1, 1], [(-w[0]/w[1])*-1 + b/w[1], (-w[0]/w[1])*1 + b/w[1]], '-', color='green', alpha=0.5, linewidth=1.0/norm)    
                    # also draw a band around the line
                    
                    band_width = 0.5/norm
                    ax.fill_between([-1, 1], [(-w[0]/w[1])*-1 + b/w[1] - band_width, (-w[0]/w[1])*1 + b/w[1] - band_width],
                                    [(-w[0]/w[1])*-1 + b/w[1] + band_width, (-w[0]/w[1])*1 + b/w[1] + band_width],
                                    color='green', alpha=0.2)   
        # Flip y axis so it's like image
        ax.invert_yaxis()
        if True:#not self._lims_set:
            ax.set_xlim(np.array(x_lim))
            ax.set_ylim(np.array(y_lim))
        

    def _init_model(self):
        """
        Initialize the TF model, either from scratch.
        Layers:
            input (2)
            circular (n_c) + linear (n_l) + sigmoid (n_s) units
            concatenation layer (n_c + n_l + n_r)
            color unit layer (n_hidden)
            output (3)

        """
        PAR_TYPES = {2: '2_param', 3: '3_param'}

        # Input is just the (x,y) pixel coordinates (scaled)
        input = Input((2,))

        div_layers = []
        for div_type, n_div in self.n_div.items():
            if n_div > 0:
                extra_kwargs = {} if div_type != 'linear' else {'parameterization':PAR_TYPES[self._line_params]}
                layer = DIV_TYPES[div_type](n_div, sharpness=self.sharpness,
                                            grad_sharpness=self.grad_sharpness, name="%s_div_layer" % (div_type,), **extra_kwargs)(input)
                div_layers.append(layer)
        if len(div_layers) == 0:
            raise Exception("Need at least one division unit (circular, linear, or sigmoid).")
        if len(div_layers) > 1:
            concat_layer = tf.keras.layers.Concatenate()(div_layers)
        else:
            concat_layer = div_layers[0]

        
        if self.n_structure>0:  # using structure layer before colors?
            structure_layer = Dense(self.n_structure, activation=tf.nn.relu, use_bias=True,
                                    kernel_initializer='random_normal', name='structure_layer')(concat_layer)
        else:
            structure_layer = concat_layer
        
        color_layer = Dense(self.n_hidden, activation=tf.nn.tanh, use_bias=True,
                            kernel_initializer='random_normal',name='color_layer')(structure_layer)
        output = Dense(3, use_bias=True, activation=tf.nn.sigmoid, name='Output_RGB')(color_layer)
        model = Model(inputs=input, outputs=output)
    
        return model

    def get_unweighted_loss(self):
        loss = self._model.evaluate(self._input, self._output, batch_size=8192, verbose=0)
        logging.info("Current loss at cycle %i:  %.6f" % (self.cycle, loss))
        return loss

    def _update_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
        self._model.optimizer.learning_rate.assign(self._learning_rate)
        logging.info("Updated learning rate to:  %.6f" % (self._learning_rate,))

    def train_more(self, epochs, learning_rate=None, noise_temps=None, verbose=True):
        if learning_rate is not None and learning_rate != self._learning_rate:
            self._update_learning_rate(learning_rate)

        input, output = self._input, self._output

        # Save numpy training set:
        # np.savez_compressed("training_data.npz", input=input, output=output, img_shape=self.image_train.shape)
        self.anneal_temp = noise_temps[0] if noise_temps is not None else 0.0
        if noise_temps is not None:
            # Langevin dynamics noise:
            noise_sds = np.sqrt(2 * learning_rate * np.array(noise_temps))

        rand = np.random.permutation(input.shape[0])
        input = input[rand]
        output = output[rand]
        # Apply the same permutation to sample weights to keep them aligned with input/output
        sample_weights = self._sample_weights[rand] if self._sample_weights is not None else None
        batch_losses = BatchLossCallback()
        swt = ", sample weight range [%.6f, %.6f]" % (np.min(sample_weights), np.max(sample_weights)) if sample_weights is not None else ""
        logging.info("... More training with %i epochs%s" % (epochs, swt))
        self._model.fit(input, output, epochs=epochs, sample_weight=sample_weights,
                        batch_size=self.batch_size, verbose=verbose, callbacks=[batch_losses])
        # Get loss for each step
        loss_history = batch_losses.losses
        self.cur_loss = np.mean(loss_history[-1]) if len(loss_history) > 0 else -2

        logging.info("Batch losses for cycle %i has %i epochs, each with %i minibatches" % (self.cycle, len(loss_history), len(loss_history[0])))
        self.cycle += 1
        
        return loss_history

    def gen_image(self, output_shape, border=0.0, keep_aspect=True, div_color_f=None, div_thickness=None):

        x, y = make_input_grid(output_shape, resolution=1.0, border=border, keep_aspect=keep_aspect)
        shape = x.shape
        logging.info("Making display image with shape:  %s" % (shape,))
        inputs = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        logging.info("Rescaled inputs to span [%.3f, %.3f] and [%.3f, %.3f], %i total samples." % (np.min(inputs[:, 0]),
                                                                                                   np.max(inputs[:, 0]),
                                                                                                   np.min(inputs[:, 1]),
                                                                                                   np.max(inputs[:, 1]),
                                                                                                   inputs.shape[0]))
        rgb = self._model.predict(inputs, batch_size=65536, verbose=True)
        logging.info("Display spans:  [%.3f, %.3f]" % (np.min(rgb), np.max(rgb)))
        img = cv2.merge((rgb[:, 0].reshape(shape[:2]), rgb[:, 1].reshape(shape[:2]), rgb[:, 2].reshape(shape[:2])))
        n_clipped = np.sum(img < 0) + np.sum(img > 1)
        logging.info("Display clipping:  %i (%.3f %%)" % (n_clipped, float(n_clipped)/img.size * 100.0))
        img[img < 0.] = 0.
        img[img > 1.] = 1.

        if div_color_f is not None:
            ### render all divider units to the image
            
            # Draw white lines on this black image, then overlay on the output image
            border_alpha_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            params = self.get_div_params()
            
            
            if 'linear' in params:
                if 'centers' in params['linear']:
                    centers = params['linear']['centers'].reshape(-1,2)  # switch to (x,y)
                    centers[:,1] = -centers[:,1]
                    angles = params['linear']['angles']

                    for i in range(self.n_div['linear']):
                        c = centers[i]
                        t = [-4.0, 4.0]
                        dx = np.cos(angles[i])
                        dy = np.sin(angles[i])
                        p0 = c + t[0] * np.array([dy, dx])
                        p1 = c + t[1] * np.array([dy, dx])
                        p0_px = self.unit_coords_to_pixels(p0.reshape(1,2), img.shape, orig_aspect=keep_aspect)[0]
                        p1_px = self.unit_coords_to_pixels(p1.reshape(1,2), img.shape, orig_aspect=keep_aspect)[0]
                        cv2.line(border_alpha_mask, (p0_px[0], p0_px[1]), (p1_px[0], p1_px[1]),
                                 color=255, thickness=div_thickness if div_thickness is not None else 1, lineType=cv2.LINE_AA)
                        
                    

                elif 'offsets' in params['linear']:
                    pass
                    # # 2 parameterization of the line (angle, offset from origin)
                    # self._artists['linear']['center_points']= None  # Unused for 2-parameter lines
                    offsets = params['linear']['offsets']
                    angles = params['linear']['angles'] 
                    normals = np.vstack((np.cos(angles), -np.sin(angles))).T
                    for l_i in range(self.n_div['linear']):
                        n = normals[l_i]
                        d = offsets[l_i]
                        # point on line closest to origin:
                        c = n * d
                        t = [-4.0, 4.0]
                        dx = np.cos(angles[l_i])
                        dy = np.sin(angles[l_i])
                        p0= c + t[0] * np.array([dy, dx])
                        p1= c + t[1] * np.array([dy, dx])
                        
                        p0_px = self.unit_coords_to_pixels(p0.reshape(1,2), img.shape, orig_aspect=keep_aspect)[0]
                        p1_px = self.unit_coords_to_pixels(p1.reshape(1,2), img.shape, orig_aspect=keep_aspect)[0]
                        points = np.array((p0_px, p1_px))

                        cv2.line(border_alpha_mask, points[0], points[1], color=255,
                                 thickness=div_thickness if div_thickness is not None else 1, lineType=cv2.LINE_AA)
                        
                    
            if 'circular' in params:
                centers = params['circular']['centers']# switch to (x,y)
                centers[:,1] = -centers[:,1]
                radii = params['circular']['radii']

                for i in range(centers.shape[0]):
                    c = centers[i]
                    r = radii[i]
                    centers_px = self.unit_coords_to_pixels(c.reshape(1,2), img.shape, orig_aspect=keep_aspect)[0]
                    r_px = self.radius_to_pixel_radius(r, img.shape, orig_aspect=keep_aspect)
                    cv2.circle(border_alpha_mask, (centers_px[0], centers_px[1]), r_px,
                               color=255, thickness=div_thickness if div_thickness is not None else 1, lineType=cv2.LINE_AA)
                    
            if 'sigmoid' in params:
                raise NotImplementedError("Rendering sigmoid units not implemented yet.")
            border_alpha_mask = border_alpha_mask[::-1]
            # Apply border mask, merge the color to the image where the mask is set, etc.
            float_mask = border_alpha_mask.astype(img.dtype) / 255.0
           
            float_mask = np.repeat(float_mask[:, :, np.newaxis], 3, axis=2)
            border_rgb = np.ones_like(img, dtype=np.float32) * np.array(div_color_f).reshape(1,1,3)
            imgb = img.copy()
            imgb = imgb * (1.0 - float_mask) + border_rgb * float_mask
            img = imgb

        return img

    def save_state(self, model_filename):
        """
        Save only the necessary args in a dict 
        """
        weights = self._model.get_weights()
        logging.info("Saving model weights with %i layers to:  %s" % (len(weights), model_filename))
        data = {'weights': weights,
                'image_raw': self.image_raw,
                'cycle': self.cycle,
                'downscale': self._downscale,
                'n_div': self.n_div,
                'n_structure': self.n_structure,
                'n_hidden': self.n_hidden,
                'sharpness': self.sharpness,
                'grad_sharpness': self.grad_sharpness
                }
        
        with open(model_filename, 'wb') as outfile:
            cp.dump(data, outfile, protocol=cp.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_state(model_filepath):
        """
        Load a saved state from a file
        :param model_filepath: path to the saved state file
        :return: the model state dict
        """
        if not os.path.exists(model_filepath):
            raise Exception("State file doesn't exist:  %s" % (model_filepath,))
        with open(model_filepath, 'rb') as infile:
            state = cp.load(infile)
        logging.info("Loaded model state from:  %s, n_weight_layers: %i, at cycle: %i" %
              (model_filepath, len(state['weights']), state['cycle']))

        return state


class BatchLossCallback(Callback):
    """
    Get loss vector (for all minibatches in an epoch) for every epoch
    """

    def __init__(self):
        super().__init__()
        self.losses = []  # list for each epoch of all minibatch losses
        
    def on_epoch_begin(self, epoch, logs=None):
        self.losses.append([])  # new epoch    

    def on_train_batch_end(self, batch, logs=None):
        self.losses[-1].append(logs['loss'])
        

class UIDisplay(object):

    """
    Show image evolving as the model trains.
    After epochs_per_cycle epochs, save the model and write an output image, update display.

    If resuming from a saved state, continue the run number from the saved state.


    """
    
    _CUSTOM_LAYERS = {'CircleLayer': CircleLayer, 'LineLayer': LineLayer, 'NormalLayer': NormalLayer}
    # Variables possible for kwargs, use these defaults if missing from kwargs

    def __init__(self, state_file=None, image_file=None, just_image=None, border=0.0, frame_dir=None, run_cycles=0,batch_size=32,center_weight_params=None, line_params=3,
                 epochs_per_cycle=1, display_multiplier=1.0, downscale=1.0,  n_div={}, n_hidden=40, n_structure=0, learning_rate=1.0, learning_rate_final=None, nogui=False, 
                 synth_image_name = None,verbose=True, div_render_params=None, anneal_args=None, **kwargs):
        self._verbose = verbose
        self._border = border
        self._epochs_per_cycle = epochs_per_cycle
        self._display_multiplier = display_multiplier
        self.n_div = n_div
        self.n_hidden = n_hidden
        self.n_structure = n_structure
        self._center_weight_params = center_weight_params
        self._update_plots = False
        self._run_cycles = run_cycles
        self._shutdown = False
        self._line_params = line_params
        self._frame_dir = frame_dir
        self._cycle = 0  # epochs_per_cycle epochs of training and an output update increments this
        self._annotate = False
        self._learn_rate = learning_rate  # updates as we anneal
        self._downscale = downscale  # constant downscale factor for training image
        self._learning_rate_init = learning_rate
        self._learning_rate_final = learning_rate_final
        self._output_image = None
        self._loss_history = [] # list {'cycle': cycle_index,'epochs': [list of minibatch losses for each epoch]}
        self._l_rate_history = []
        self._anneal_history = []
        self._hist_lock = Lock()  # for history lists
        self._batch_size = batch_size
        self._nogui = nogui
        self._metadata = []
        self.final_loss = None
        self._show_dividers = True
        self._ui_flags = {n:False for n in range(10)}  # for keypress toggles
        self.div_render_params = div_render_params if div_render_params is not None else {}
        self._anneal = anneal_args
        self._show_all_history = True
        self.curr_loss_unweighted = -1
        self.last_epoch_mean_loss = -1
        self._kwargs = kwargs  # save for later if needed
        
        if learning_rate_final is not None:
            if learning_rate_final < 0:
                raise Exception("Final learning rate must be non-negative for annealing.")
            if self._run_cycles <= 0:
                raise Exception("Must specify run_cycles > 0 if annealing with learning_rate_final.")
            if learning_rate >0:
                self._learning_rate_decay = (learning_rate_final / learning_rate) ** (1.0 / (self._run_cycles-1)) if self._run_cycles > 1 else 1.0
                logging.info("Using learning rate annealing:  initial: %.6f, final: %.6f, decay: %.6f per cycle over %i cycles (Decay constant: %.6f)." %
                            (learning_rate, learning_rate_final, self._learning_rate_decay, self._run_cycles, self._learning_rate_decay))
            else:
                self._learning_rate_decay = 0.0
        else:
            self._learning_rate_decay = None
            
        self._image_raw, self._file_prefix = self._set_image(image_file, synth_image_name)
        
        self._sim = ScaleInvariantImage(n_div=self.n_div, n_hidden=n_hidden, n_structure=n_structure, learning_rate_initial=self._learn_rate,
                                        batch_size=self._batch_size, state_file=state_file, image_raw=self._image_raw, line_params=self._line_params,
                                        downscale=self._downscale, center_weight_params=self._center_weight_params, **kwargs)

        # Check for metadata file
        if state_file is not None:
            meta_filename = self.get_filename('metadata')
            if os.path.exists(meta_filename):
                with open(meta_filename, 'r') as f:
                    metadata = json.load(f)
                logging.info("Loaded metadata from:  %s" % (meta_filename,))
                self._metadata = metadata['frames']  if 'frames' in metadata else metadata
                
                if 'anneal_history' in metadata:
                    self._anneal_history = metadata['anneal_history'] 
                    logging.info("Loaded anneal history with %i entries." % (len(self._anneal_history),))
                else:
                    logging.warning("Metadata found but contains no anneal history")
                    self._anneal_history = []
                
                if 'loss_history' in metadata:
                    self._loss_history = metadata['loss_history'] 
                    logging.info("Loaded loss history with %i entries." % (len(self._loss_history),))
                else:
                    logging.warning("Metadata found but contains no loss history")
                    self._loss_history = []
                    
                if 'learning_rate_history' in metadata:
                    self._l_rate_history = metadata['learning_rate_history'] if 'learning_rate_history' in metadata else []
                    logging.info("Loaded learning rate history with %i entries." % (len(self._l_rate_history),))
                else:
                    logging.warning("Metadata found but contains no learning rate history")
                    self._l_rate_history = []
            else:
                logging.info("No metadata file found:  %s" % (meta_filename,))

        logging.info("Using output file prefix:  %s" % (self._file_prefix,))

        self._output_shape = None  # for saving:  set during training, train_image.shape * display_multiplier, high res-generated image
        self._frame = None

        self._just_image = just_image
        
    def _set_image(self, img_filename, synth_name, image_size_wh = (2*64, 2*(32+16))):
        """
        disambiguate, load  / generate image
        :param img_filename: path to image file
        :param synth_name: name of the synthetic image type to generate, or None
        :param image_size_wh: size of synthetic image to generate (width, height)
        """
        if synth_name is not None:
            image_maker = TestImageMaker(image_size_wh)
            image_raw = image_maker.make_image(synth_name)
            file_prefix = "SYNTH_%s" % (synth_name,)
            logging.info("Generated synthetic image type %s with shape %s, using prefix %s" %
                         (synth_name, image_raw.shape, file_prefix))
        else:
            
            if not os.path.exists(img_filename):
                raise Exception("Image file doesn't exist:  %s" % (img_filename,))
            image_raw = cv2.imread(img_filename)[:, :, ::-1]

            # model/image file prefix is the image bare name without extension
            file_prefix = os.path.basename(os.path.splitext(os.path.basename(img_filename))[0])
            logging.info("Loaded image %s with shape %s, using save_prefix %s" %
                        (img_filename, image_raw.shape, file_prefix))

        return image_raw, file_prefix

    def _get_arch_str(self):
        """
        for 15 lines and 15 circles + 10 color units, should look like:
           15c-15l_10h
        """
        arch_str = ""

        for div_type in ['circular', 'linear', 'sigmoid']:
            n_div = self.n_div.get(div_type, 0)
            if n_div > 0:
                arch_str += "%i%s-" % (n_div, div_type[0])
        arch_str = arch_str[:-1]  # remove trailing -
        if self.n_structure >0:
            arch_str += "_%it" % (self.n_structure,)
        arch_str += "_%ic" % (self.n_hidden,)
        return arch_str

    def get_filename(self, which='model'):
        if which == 'model':
            return "%s_model_%s.pkl" % (self._file_prefix, self._get_arch_str())
        elif which == 'frame':
            return "%s_output_%s_cycle-%.8i.png" % (self._file_prefix, self._get_arch_str(), self._sim.cycle)
        elif which == 'single-image':
            return "%s_single_%s.png" % (self._file_prefix, self._get_arch_str())
        elif which == 'train-image':
            return "%s_train_%s_downscale=%.1f.png" % (self._file_prefix, self._get_arch_str(), self._downscale)
        elif which == 'metadata':
            return "%s_metadata_%s.json" % (self._file_prefix, self._get_arch_str())
        else:
            raise Exception("Unknown filename type:  %s" % (which,))

    def _save_training_image(self):
        filename = self.get_filename('train-image')

        # file_path = os.path.join(self._frame_dir, filename) if self._frame_dir is not None else filename
        file_path = filename
        train_img = self._sim.image_train
        cv2.imwrite(file_path, train_img[:, :, ::-1])
        logging.info("Wrote training image:  %s" % (file_path,))

    def _train_thread(self, max_iter=-1):
        # TF 2.x doesn't use sessions or graphs in eager

        # Print full model architecture
        self._sim._model.summary()

        logging.info("Starting training:")
        logging.info("\tBatch size: %i" % (self._sim.batch_size,))
        logging.info("\tDiv types:  %s" % (self._get_arch_str(),))
        logging.info("\tStructure units:  %i" % (self.n_structure,))
        logging.info("\tColor units:  %i" % (self.n_hidden,))
        logging.info("\tSharpness: %f" % (self._sim.sharpness,))
        logging.info("\tGradient sharpness: %f" % (self._sim.grad_sharpness,))

        self._cycle = self._sim.cycle  # in case resuming from a saved statese\\
        run_cycle = 0  # number of cycles since starting this run (not counting previous runs if resuming from a saved state)

        if self._run_cycles > 0:
            logging.info("Will run for %i cycles of %i epochs each (total %i epochs), starting from cycle %i." %
                         (self._run_cycles, self._epochs_per_cycle, self._run_cycles * self._epochs_per_cycle, self._cycle))

        # write frame before any training (may overwrite last frame if continuing a run).
        self._output_image = self._gen_image()
        logging.info("Generated output image of shape:  %s" % (self._output_image.shape,))       
        self._save_training_image()
        # if we're continuing, don't store the new output image
        
        ### DEBUG TEMP
        cv2.imwrite("DEBUG_initial_image.png", (self._output_image * 255).astype(np.uint8)[:, :, ::-1])

        if self._cycle == 0:
            # Initial image, random weights, no training.  (Remove?)
            cur_loss_uw = self._sim.get_unweighted_loss()
            frame_name = self._write_frame(self._output_image)  # Create & save image
            init_meta = {'cycle': self._sim.cycle,
                         'learning_rate': self._learn_rate,
                         'current_loss': cur_loss_uw,
                         'filename': frame_name}
            self._metadata.append(init_meta)
            self._write_metadata()

        anneal_temp, anneal_decay = 0, 0
        if self._anneal is not None:
            anneal_temp = self._anneal[0]
            if self._run_cycles == 0:
                anneal_decay = self._anneal[1]  # if running forever, just use the decay constant
            else:
                n_epochs = self._epochs_per_cycle * (self._run_cycles if self._run_cycles > 0 else 1)
                anneal_decay = (self._anneal[1]/self._anneal[0]) ** (1.0 / n_epochs)  # decay per epoch
                logging.info("Using Langevin dynamics with initial temp %.6f, decay %.6f per epoch over %i epochs." %
                            (anneal_temp, anneal_decay, n_epochs))

        self.curr_loss_unweighted = -1.0
        while (max_iter==-1 or run_cycle < max_iter) and not self._shutdown and (run_cycle < self._run_cycles or self._run_cycles == 0):
            if self._shutdown:
                break
            logging.info("Training batch_size: %i, cycle: %i of %i, learning_rate: %.6f" %
                         (self._sim.batch_size, self._sim.cycle, self._run_cycles, self._learn_rate))
            
            if self._anneal is not None:
                anneal_temps = anneal_temp * (anneal_decay ** np.arange(self._epochs_per_cycle))
                logging.info("Using Langevin dynamics with noise temps:  %.6f, ..., %.6f" % (anneal_temps[0], anneal_temps[-1]))
                anneal_temp = anneal_temps[-1]  * anneal_decay  # update for next cycle
            else:
                anneal_temps = np.zeros(self._epochs_per_cycle)
            
            new_losses = self._sim.train_more(self._epochs_per_cycle, 
                                              learning_rate=self._learn_rate,
                                              verbose=self._verbose, 
                                              noise_temps=anneal_temps)
            cur_loss_uw = self._sim.get_unweighted_loss()
            last_epoch_mean_loss = np.mean(new_losses[-1]) if len(new_losses) > 0 else -1
            output_image = self._gen_image()
            
            
            with self._hist_lock:
                
                
                self.curr_loss_unweighted = cur_loss_uw
                self.last_epoch_mean_loss = last_epoch_mean_loss    
                self._output_image = output_image
                self._l_rate_history.append(self._learn_rate)
                self._anneal_history.append(anneal_temps.tolist())
                self._loss_history.append({'cycle': self._cycle, 'epochs': new_losses,'final_loss': cur_loss_uw})
                
            self._cycle = self._sim.cycle  # update AFTER saving data, since train_more increments it at the end

            frame_name = self._write_frame(self._output_image)  # Create & save image
            self._metadata.append({'cycle': self._sim.cycle, 'learning_rate': self._learn_rate, 'current_loss': cur_loss_uw, 'filename': frame_name})
            self._write_metadata()
            filename = self.get_filename('model')
            out_path = filename  # Just write to cwd instead of os.path.join(img_dir, filename)
            self._sim.save_state(out_path)  # TODO:  Move after each cycle
            logging.info("Saved model state to:  %s" % (out_path,))
            
            if self._shutdown:
                break
            
            run_cycle += 1
            

            if run_cycle >= self._run_cycles and self._run_cycles > 0:
                logging.info("Reached max cycles (%i), stopping." % (self._run_cycles,))
                self._shutdown = True
                break

            if self._learning_rate_decay is not None:
                self._learn_rate = self._learning_rate_decay * self._learn_rate
                logging.info("Decayed learning rate to:  %.6f" % (self._learn_rate,))

        # print FFMPEG command to make a movie from the images
        if self._frame_dir is not None:
            logging.info("To make a movie from the images, try:")
            logging.info("  ffmpeg -y -framerate 10 -i %s_output_%s_cycle-%%08d.png -c:v libx264 -pix_fmt yuv420p output_movie.mp4" %
                         (os.path.join(self._frame_dir, self._file_prefix), self._get_arch_str()))
        self.final_loss = cur_loss_uw
        logging.info("Final loss after %i cycles:  %.6f" % (run_cycle, cur_loss_uw))
        return cur_loss_uw

    def _gen_image(self, shape=None):
        if shape is None:
            train_shape = self._sim.image_train.shape
            shape = (np.array(train_shape[:2]) * self._display_multiplier).astype(int)
        img = self._sim.gen_image(output_shape=shape, border=self._border, **self.div_render_params)
        return img

    def _write_image(self, img, filename=None):
        out_filename = self.get_filename('single-image') if filename is None else filename
        cv2.imwrite(out_filename, np.uint8(255 * img[:, :, ::-1]))
        logging.info("Wrote:  %s" % (out_filename,))

    def _write_frame(self, img):
        out_path = None
        
        if self._frame_dir is not None:
            if not os.path.exists(self._frame_dir):
                os.makedirs(self._frame_dir)
                logging.info("Created frame directory:  %s" % (self._frame_dir,))
            out_filename = self.get_filename('frame')
            out_path = os.path.join(self._frame_dir, out_filename)
            if os.path.exists(out_path):
                logging.warning("Output image %s already exists, overwriting!!!!" % (out_path,))
            cv2.imwrite(out_path, np.uint8(255 * img[:, :, ::-1]))
            logging.info("Wrote:  %s" % (out_path,))
        return out_path
    
    def _write_metadata(self):
        # # only if saving frames
        # if self._frame_dir is not None:
        # write metadata to same file, just overwrite:
        meta_filename = self.get_filename('metadata')
        meta_path = '.'
        meta_file_path = os.path.join(meta_path, meta_filename)
        metadata = {'frames': deepcopy(self._metadata),
                    'model_file': os.path.abspath(self.get_filename('model')),
                    'train_image_file': os.path.abspath(self.get_filename('train-image')),
                    'train_downscale': self._downscale,
                    'loss_history': self._loss_history,
                    'learning_rate_history': self._l_rate_history,
                    'anneal_history': self._anneal_history}
        with open(meta_file_path, 'w') as f:
            json.dump(metadata, f)
        logging.info("Wrote METADATA file to --------> :  %s" % (meta_file_path,))

    def _start(self):
        
        self._worker = Thread(target=self._train_thread)
        self._worker.start()
        logging.info("Started worker thread.")

    def _keypress_callback(self, event):
        if event.key == 'x' or event.key == 'escape' or event.key == 'q':
            logging.info("Shutdown requested, waiting for worker to finish...")
            self._shutdown = True
        elif event.key =='d':
            # toggle annotation of divider units
            self._show_dividers = not self._show_dividers
            logging.info("Toggled divider unit annotation:  %s" % ("on" if self._show_dividers else "off",))
            self._update_plots = True
        elif event.key == 'a':
            self._show_all_history= not self._show_all_history
            logging.info("Toggled showing all loss history:  %s" % ("on" if self._show_all_history else "off",))
            self._update_plots = True
        elif event.key == 'w':
            self._show_weight_contours = not self._show_weight_contours
            logging.info("Toggled weight contour display:  %s" % ("on" if self._show_weight_contours else "off",))   
            self._update_plots = True
        elif event.key in [str(n) for n in range(10)]:
            n = int(event.key)
            self._ui_flags[n] = not self._ui_flags[n]
            logging.info("Toggled UI flag %i to:  %s" % (n, "on" if self._ui_flags[n] else "off"))
            self._update_plots = True
        else:
            logging.info("Unassigned key: %s" % (event.key,))
        
    def get_train_image(self):
        return self._sim.image_train

    def run(self, debug_epochs_nothread=-1):
        """
        Run as interactive matplotlib window, updating when new image is available.
           * left is the current training image
           * middle is the current output image
           * right is the loss history
        :param debug_epochs_nothread: if > -1, run this many epochs in the main thread instead of a worker thread (for debugging)
        """
        if self._just_image is not None:
            img_shape = (np.array(self._sim.image_raw.shape[:2]) * self._display_multiplier).astype(int)
            img = self._gen_image(shape=img_shape)
            self._write_image(img, filename = self._just_image)
            return

        if debug_epochs_nothread>-1:
            logging.info("Debug mode, running training in main thread.")
            
            loss = self._train_thread(max_iter = debug_epochs_nothread)
            self._output_image = self._gen_image()
            #fig, ax = plt.subplots(1,1)
            #self._sim.draw_div_units(ax=ax, output_image=self._output_image, plot_units=True, draw_flags=self._ui_flags)
            # plt.show()
            self._worker = None
        else:
            self._start()

        if self._nogui:
            logging.info("No GUI mode, waiting for training to finish...")
            if self._worker is not None:
                self._worker.join()
            return self.final_loss, self._output_image

        while self._sim.image_train is None:
            time.sleep(0.05)

        plt.ion()            
        fig = plt.figure(figsize=(12,8))

        if self._image_raw.shape[0] > self._image_raw.shape[1]:
            # tall images, side-by side, plots on the right
            
            #
            #   +------+ +------+ +--+
            #   |train | |output| |p1|
            #   |      | |      | +--+
            #   |      | |      | +--+
            #   |      | |      | |  |
            #   |      | |      | |p2|
            #   |      | |      | |  |
            #   +------+ +------+ +--+

            grid = gridspec.GridSpec(8, 3, width_ratios=[1,1,1])
            logging.info("Tall images, orienting side-by-side.")
            train_ax = fig.add_subplot(grid[:, 0])
            out_unit_ax= fig.add_subplot(grid[:, 1])
        else:
            # wide images, stacked vertically, plots on the right
            #
            #   +-------------+ +--+
            #   | TRAIN       | |p1|
            #   |             | +--+
            #   +-------------+ +--+ 
            #   +-------------+ |  |
            #   |             | |p2|
            #   | Output      | |  |
            #   +-------------+ +--+
            logging.info("--------------------------Wide images, orienting vertically.")
            grid = gridspec.GridSpec(8, 2, width_ratios=[2,1])
            train_ax = fig.add_subplot(grid[:4, 0])
            out_unit_ax = fig.add_subplot(grid[4:, 0])
        
        #if self._center_weight_params is not None: 
        if self._anneal is None:           
            lrate_ax = fig.add_subplot(grid[:3, -1])
            loss_ax = fig.add_subplot(grid[3:, -1],sharex=lrate_ax)
            anneal_ax = None
        else:
            lrate_ax = fig.add_subplot(grid[:2, -1])
            anneal_ax = fig.add_subplot(grid[2:4, -1],sharex=lrate_ax)
            loss_ax = fig.add_subplot(grid[4:, -1],sharex=lrate_ax)
        
        weights_plotted=False
        
        lrate_ax.grid(which='major', axis='both')
        loss_ax.grid(which='both', axis='both')
        lrate_ax.tick_params(labeltop=False, labelbottom=False)# turn off x tick labels for lrate
        lrate_ax.set_ylabel("Learning Rate")
        train_ax.axis("off")        
        #out_unit_ax.axis("off")
        # turn off x ticks, ylabel & ticks for out_unit_ax
        out_unit_ax.tick_params(labeltop=False, labelbottom=False)
        # out_unit_ax.set_ylabel("")
        # out_unit_ax.set_xticks([])
        # out_unit_ax.set_yticks([])
        cmd = "hit 'a' to show last 5 cycles only." if self._show_all_history else "Hit 'a' to show all cycles."
        loss_ax.set_xlabel("Cycle (%i epochs) - %s" % (self._epochs_per_cycle, cmd))
        loss_ax.set_ylabel("Loss")

        loss_ax.set_title("Loss history, %i cycles total" % (len(self._loss_history),), fontsize=10)
        if anneal_ax is not None:
            anneal_ax.grid(which='major', axis='both')
            anneal_ax.set_ylabel("Temp", fontsize=8)
            anneal_ax.set_title("Annealing Temp:  %.6f" % (self._sim.anneal_temp,), fontsize=10)

        lrate_ax.set_ylabel("")  # Learning Rate")
        lrate_title = "Learning Rate, current %.6f" % (self._learn_rate,)
        lrate_title += "\nCurrent Annealing Temp:  %.6f" % (self._sim.anneal_temp,) if self._sim.anneal_temp > 0 else ""
        lrate_ax.set_title(lrate_title, fontsize=12)
        lrate_ax.set_yscale('log')
        
        # Architecture string, for titles
        div_units_str = ""
        if self.n_div['circular'] > 0:
            div_units_str += "%i Circle units " % (self.n_div['circular'],)
        if self.n_div['linear'] > 0:
            div_units_str += "%i Line units (lines use the %i-parameterization)" % (self.n_div['linear'], self._line_params)
        if self.n_div['sigmoid'] > 0:
            div_units_str += "%i Sigmoid units" % (self.n_div['sigmoid'], )

        fig.canvas.mpl_connect('key_press_event', self._keypress_callback)

        artists = {'loss': None, 'train_img': None, 'out_img': None}
        self._update_plots = False # set when user changes something that requires a plot update
        self._show_weight_contours = True
        # start main UI loop (training is in the thread):
        graph_update_cycle = -1  # last cycle we updated the graph, don't change unless it changes
        cleanup_counter = 0  # Track when to do matplotlib cleanup
        
        
        # Just draw training image once:
        img_out = self._sim.image_train.copy()
        if self._center_weight_params is not None and self._sim._weight_grid is not None and self._show_weight_contours:
            # apply contour lines at 20% intervals
            n_cont = 7
            contour_levels = [measure.find_contours(self._sim._weight_grid,level = l) for l in np.linspace(1.0, self._center_weight_params['w_max'], n_cont, endpoint=True)[1:]]
            colors = plt.cm.viridis(np.linspace(0,1,len(contour_levels)))[:,:3]
            for level_ind, contours in enumerate(contour_levels):
                for contour in contours:
                    contour = np.round(contour).astype(int)
                    img_out[contour[:, 0], contour[:, 1], :] = colors[level_ind]
            logging.info("Overlayed weight contours on training image.")
        artists['train_img'] = train_ax.imshow(img_out)
        train_ax.set_anchor('C')

        train_h, train_w = self._sim.image_train.shape[:2]
        box_aspect = (train_h / float(train_w)) if train_w else 1.0
        if hasattr(train_ax, "set_box_aspect"):
            train_ax.set_box_aspect(box_aspect)
            out_unit_ax.set_box_aspect(box_aspect)
        else:
            train_ax.set_aspect(box_aspect, adjustable='box')
            out_unit_ax.set_aspect(box_aspect, adjustable='box')
        LPT.reset(enable=False, burn_in=4, display_after=100,save_filename = "loop_timing_log.txt")
        plt.tight_layout()
        tl_times=[]
        while not self._shutdown and (self._worker is None or self._worker.is_alive()):
            # try:
            LPT.mark_loop_start()

            # Update titles w/dynamic info every loop 
            cmd =( "\n('w' to hide weight contours)" if self._show_weight_contours else "\n('w' to show weight contours)")\
                if self._center_weight_params is not None else ""
            train_ax.set_title("Training cycle %i/%s, target image %i x %i%s" %
                (self._cycle+1, self._run_cycles if self._run_cycles > 0 else '--',
                    self._sim.image_train.shape[1], self._sim.image_train.shape[0], cmd))
            
            loss_ax.set_title("Training Loss History\n1 dot = 1 minibatch (%i samples)" % (  self._batch_size), fontsize=12)
            if anneal_ax is not None:
                anneal_ax.set_title("Annealing Temp:  %.6f" % (self._sim.anneal_temp,), fontsize=12)
            lrate_title = "Learning Rate, current %.6f" % (self._learn_rate,)
            lrate_ax.set_title(lrate_title, fontsize=12)
            cmd = "hit 'a' to show last 5 cycles only." if self._show_all_history else "Hit 'a' to show all cycles."
            loss_ax.set_xlabel("Cycle (%i epochs) - %s" % (self._epochs_per_cycle, cmd))

            cmd = "(Hit 'd' to hide division units.)" if self._show_dividers else "(Hit 'd' to plot division units.)"
            out_unit_ax.set_title("Output image %s" % (cmd,), fontsize=12)
            last_epoch_loss =  self.last_epoch_mean_loss
            cur_loss = self.curr_loss_unweighted
            wt = " (weighted)" if self._center_weight_params is not None else ""
            loss_str = "   Last cycle mean loss:  %.6f\nLast training epoch loss%s: %.6f" % (cur_loss, wt,last_epoch_loss)
            
            out_unit_ax.set_xlabel(loss_str, fontsize=12)
            


            # Update output image & draw divider units
            if self._output_image is not None:
                out_img =self._output_image
                self._sim.draw_div_units(out_unit_ax, output_image=out_img, plot_units=self._show_dividers)            
                LPT.add_marker("output image done")

            
            if (len(self._loss_history) > 0 and len(self._l_rate_history )> 0 and \
                self._cycle != graph_update_cycle) or self._update_plots:
                """
                X-axis will count CYCLES. 
                
                For the loss history, plot each minibatch as a dot, the epoch means as red line segments, and the cycle means as yellow lines
                Show the final loss as a yellow dot at the end of each cycle.
                """
                if len(self._loss_history) ==0 or len(self._l_rate_history) == 0:
                    self._update_plots = False
                    continue
                graph_update_cycle = self._cycle
                # Update titles
                
                # Prepare data
                first_cycle = 0 if self._show_all_history else max(0, self._cycle - 5)
                
                with self._hist_lock:
                    
                    loss_history = deepcopy(self._loss_history[first_cycle:])
                    l_rate_history = deepcopy(self._l_rate_history[first_cycle:])
                    anneal_history = deepcopy(self._anneal_history[first_cycle:]) if self._anneal is not None else None
                    
                    
                    
                epoch_means = []
                epoch_means_x = []
                cycle_means = []
                cycle_means_x = []                
                minibatch_x = []
                minibatch_losses = []
                lrate_x = []
                lrate_y = []
                anneal_temp_x = []
                anneal_temp_y = []
                for cyc in range(len(loss_history)):
                    cyc_ind = first_cycle + cyc
                    n_epochs = len(loss_history[cyc]['epochs'])
                    
                    # x-coord for start of each epoch this cycle:
                    epoch_x = np.linspace(0, 1, n_epochs, endpoint=False) + cyc_ind
                    epoch_dx = 1.0 / n_epochs
                    
                    # scalar
                    lrate_x.append(cyc_ind)
                    lrate_y.append(l_rate_history[cyc])
                    
                    if anneal_history is not None:
                        y = anneal_history[cyc]
                        if n_epochs != len(y):
                            raise Exception("Cycle %i has %i epochs but anneal history has %i entries!" % (cyc_ind, n_epochs, len(y)))
                        anneal_temp_x.extend(epoch_x.tolist())
                        anneal_temp_y.extend(y)
                    e_means = []
                    
                    for ep in range(n_epochs):
                        mbh = loss_history[cyc]['epochs'][ep]
                        mbh_x = np.linspace(0, epoch_dx, len(mbh), endpoint=False) + epoch_x[ep]
                        minibatch_x.extend(mbh_x.tolist())
                        minibatch_losses.extend(mbh)
                        e_means.append(np.mean(mbh))
                        
                    epoch_means.extend(e_means)
                    epoch_means_x.extend(epoch_x.tolist())
                    cycle_means.append(np.mean(e_means))
                    cycle_means_x.append(cyc_ind)

                def _make_flat_line_segments(x0, x1, y):
                    x = np.zeros(2 * len(y))
                    y_flat = np.zeros(2 * len(y))
                    x[0::2] = x0
                    x[1::2] = x1
                    y_flat[0::2] = y
                    y_flat[1::2] = y
                    return x, y_flat


                # Plot / update loss axis
                
                ex, ey = _make_flat_line_segments(epoch_means_x, epoch_means_x[1:] + [cyc_ind+1], epoch_means)
                cx, cy = _make_flat_line_segments(cycle_means_x, np.concatenate((cycle_means_x[1:],[cyc_ind+1])), cycle_means)
                if artists['loss'] is None:
                    artists['loss'] = {'minibatch_data': loss_ax.plot(minibatch_x, minibatch_losses, 'b.', label='Minibatch Loss')[0],
                                       'epoch_means': loss_ax.plot(ex, ey, 'r-', label='Epoch Means')[0],
                                       'cycle_means': loss_ax.plot(cx, cy, '-', color=np.array((8,255,8))/255.0, label='Cycle Means')[0]}
                    loss_ax.legend(loc='upper right')
                    # Set y to log-scale
                    loss_ax.set_yscale('log')
                    
                else:
                    artists['loss']['minibatch_data'].set_data(minibatch_x, minibatch_losses)
                    artists['loss']['epoch_means'].set_data(ex, ey)
                    artists['loss']['cycle_means'].set_data(cx, cy)
                # set x and y limits (x should be flush to cycle boundaries, y should have a .05 margin top and bottom)
                x_min, x_max = cycle_means_x[0], cycle_means_x[-1]
                y_min, y_max = np.min(minibatch_losses), np.max(minibatch_losses)
                loss_ax.set_xlim(x_min - 0.025 * (x_max - x_min), x_max + 0.025 * (x_max - x_min))
                loss_ax.set_ylim(y_min/10.0**.1, y_max*10.0**.1)
                
                LPT.add_marker("loss plot done")
                # For each learning rate / cycle pair we need to plot a horizontal line segment, so make the list twice as long
                # and plot each y value twice, advancing x by 1 betweeen the first and second copy of each y value.

                lrate_x, lrate_y = _make_flat_line_segments(lrate_x, lrate_x[1:] + [cyc_ind+1], lrate_y)
                if artists.get('lrate') is None:
                    artists['lrate'] = lrate_ax.plot(lrate_x, lrate_y, 'r-', label='Learning Rate')[0]
                    lrate_ax.set_yscale('log')
                else:
                    artists['lrate'].set_data(lrate_x, lrate_y)
                y_min, y_max = np.min(lrate_y)*.95, np.max(lrate_y)*1.05 
                lrate_ax.set_ylim(y_min, y_max)
                
                if anneal_ax is not None:
                    anneal_temp_x, anneal_temp_y = np.array(anneal_temp_x), np.array(anneal_temp_y)
                    if artists.get('anneal') is None and anneal_history is not None:
                        artists['anneal'] = anneal_ax.plot(anneal_temp_x, anneal_temp_y, label='Anneal Temp')[0]
                    elif artists.get('anneal') is not None and anneal_history is not None:
                        artists['anneal'].set_data(anneal_temp_x, anneal_temp_y)
                    anneal_range = np.max(anneal_temp_y) - np.min(anneal_temp_y)
                    y_min, y_max = (np.min(anneal_temp_y)-anneal_range*.05, 
                        np.max(anneal_temp_y) + anneal_range*.05)  
                    anneal_ax.set_ylim(y_min, y_max)
                
                # set shared x limit
                x_min, x_max = cycle_means_x[0], cycle_means_x[-1]
                lrate_ax.set_xlim(x_min, x_max)
            
            #Set loss rate axis x ticks to integers only:
            ticker = plt.MaxNLocator(integer=True)
            loss_ax.xaxis.set_major_locator(ticker)
            if anneal_ax is not None:
                anneal_ax.xaxis.set_major_locator(ticker)

            self._update_plots = False

            # Periodic cleanup to prevent matplotlib memory accumulation
            cleanup_counter += 1
            if False:#cleanup_counter % 50 == 0:  # Every 50 iterations
                # Clear matplotlib's internal caches
                fig.canvas.flush_events()
                # Force garbage collection of matplotlib artists
                import gc
                gc.collect()
                LPT.add_marker("cleanup done")
            t0 = time.perf_counter()                
            plt.tight_layout()
            tl_times.append(time.perf_counter() - t0)
            if len(tl_times) % 10 == 0:
                logging.info("Tight_layout time (last %i iters): min %.3f sec, max %.3f sec, mean %.3f sec" % (len(tl_times), 
                                                                                                               np.min(tl_times), 
                                                                                                               np.max(tl_times),
                                                                                                               np.mean(tl_times)))
                #tl_times=[]

            plt.draw()
            fig.canvas.flush_events()  # Force canvas to update
            plt.pause(0.2 )
            # except Exception as e:
            #     logging.error(f"Error updating GUI: {e}")
            #     plt.pause(0.1)  # Longer pause on error

            if self._shutdown:
                break
        
        # Clean up after training is done
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=5.0)
        plt.ioff()
        plt.show()  # Keep the final plot visible
        
        return self.final_loss, self._output_image


def get_args():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Learn an image.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_image", help="image to transform", type=str, default=None)
    parser.add_argument("--test_image", help="Which test image to use (internally generated, overrides --input_image)", type=str, default=None)
    
    parser.add_argument("-c", "--circles", help="Number of circular division units.", type=int, default=0)
    parser.add_argument("-l", "--lines", help="Number of linear division units.", type=int, default=0)
    parser.add_argument("-lp", "--lines_params", help="specify 2 or 3. (2 means lines are parameterized by angle and distance from origin,"+
                                                     "3 means lines are parameterized by angle and intercept / center point).", type=int, default=3)
    parser.add_argument("-s", "--sigmoids", help="Number of sigmoid division units.", type=int, default=0)
    parser.add_argument("-t", "--structure_units", help="Size of structure layer (before color layer), default 0 (no structure layer)"
                        , type=int, default=0)
    parser.add_argument("-n", "--n_hidden", help="Number of hidden_units units in the model.", type=int, default=64)
    parser.add_argument("-m", "--model_file", help="model to load & continue.", type=str, default=None)
    parser.add_argument("-j", "--just_image", help="Just generate an an image (use with -m to"+
                        "generate a high-res image from a trained model)", type=str, default=None)
    parser.add_argument("-e", "--epochs_per_cycle", help="Number of epochs between frames.", type=int, default=1)
    parser.add_argument(
        '-k', '--cycles', help="Number of training cycles to do (epochs_per_cycle epochs each). 0 = run forever.", type=int, default=0)
    parser.add_argument("-x", "--disp_mult",
                        help="Display image dimension multiplier (X training image size).", type=float, default=1.0)
    parser.add_argument(
        "-b", "--border", help="Extrapolate outward from the original shape by this factor.", type=float, default=0.0)
    parser.add_argument(
        "-p", "--downscale", help="downscale image by this factor, speeds up training at the cost of detail.", type=float, default=1.0)
    parser.add_argument('-r', "--learning_rate", help="Learning rate for the optimizer.", type=float, default=1.0)
    parser.add_argument('-a', "--learning_rate_final",
                        help="Reduce by multiplicative constant over <cycles> cycles.", type=float, default=None)
    parser.add_argument("--sharpness", help="Sharpness constant for activation function, " +
                        "activation(excitation) = tanh(excitation*sharpness).", type=float, default=1000.0)
    parser.add_argument('--gradient_sharpness', help="Use false gradient with this sharpness (tanh(grad_sharp * cos_theta)) instead of actual" +
                        " (very flat) gradient.  High values result in divider units not moving very much, too low" +
                        " and they don't settle.", type=float, default=5.0)
    parser.add_argument('-f', '--save_frames',
                        help="Save frames during training to this directory (must exist).", type=str, default=None)
    parser.add_argument("-w", "--weigh_center", 
                        help="Weigh pixels nearer the center higher during training by this r_inner, r_outer, lr_max, and x, y offsets.",
                        nargs=5, type=float, default=[])
    parser.add_argument("--nogui", help="No GUI, just run training to completion.", action='store_true', default=False)
    parser.add_argument('-z', '--batch_size', help="Training batch size.", type=int, default=32)
    parser.add_argument('-d', '--render_dividers', type=int, nargs=4, default=None, 
                    help="Generate output images with division units rendered as lines, params are THICKNESS RED GREEN BLUE (ints)")
    
    parser.add_argument('--anneal', type=float, nargs=3, default=None, help="Annealing parameters: [T_init] [T_final/decay] [n_cycles]: "+
                        "where the temperature is exponentially decayed from T_init to T_final over n_cycles (if training for longer, T=0 after n_cycles)." )
    parsed = parser.parse_args()
    
    # Assemble various arg collections:
    
    # Divider units:
    n_div = {'circular': parsed.circles, 'linear': parsed.lines, 'sigmoid': parsed.sigmoids}
    if parsed.render_dividers is not None:
        div_render = {'div_thickness': int(parsed.render_dividers[0]), 
                      'div_color_f': ((parsed.render_dividers[1]/255.0), 
                                      (parsed.render_dividers[2]/255.0), 
                                      (parsed.render_dividers[3]/255.0))}
    else:
        div_render = None
        
    # Pixel importance weights (optional):
    if len(parsed.weigh_center) == 5:
        center_weight = {'w_max': parsed.weigh_center[0], 
                         'r_inner': parsed.weigh_center[1],
                         'r_outer': parsed.weigh_center[2],  
                         'offsets_xy_rel': (parsed.weigh_center[3], parsed.weigh_center[4])}
    elif len(parsed.weigh_center) != 0:
        print(parsed.weigh_center)
        raise Exception("If using --weigh_center, must provide 5 values:  r_inner, r_outer, w_max, x_offset_rel, y_offset_rel")
    else:
        center_weight = None
        
    kwargs = {'epochs_per_cycle': parsed.epochs_per_cycle, 'display_multiplier': parsed.disp_mult, 'center_weight_params': center_weight,
              'border': parsed.border, 'sharpness': parsed.sharpness, 'grad_sharpness': parsed.gradient_sharpness,'line_params': parsed.lines_params,
              'downscale': parsed.downscale, 'n_div': n_div, 'frame_dir': parsed.save_frames, 'batch_size': parsed.batch_size,'div_render_params': div_render,
              'just_image': parsed.just_image, 'n_hidden': parsed.n_hidden, 'run_cycles': parsed.cycles, 'n_structure': parsed.structure_units,
              'learning_rate': parsed.learning_rate, 'nogui': parsed.nogui, 'learning_rate_final': parsed.learning_rate_final, 'anneal_args': parsed.anneal}
    print(parsed.anneal)
    return parsed, kwargs


def test_vertical():
    
    lines = {'centers': np.array([[0.0, 0.2], [0.0, -0.2]]),
             'angles': np.array([0, 1])}
    tim = TestImageMaker(image_size_wh=(20,36))
    test_image = tim.make_image('spec_image', lines = lines, is_color=False)
    image_filename = "test_vertical.png"
    cv2.imwrite(image_filename, np.uint8(255*test_image))
    n_div = {'linear': 2, 'circular': 0, 'sigmoid': 0}

    kwargs = {'image_file': image_filename, 'n_hidden': 4, 'n_structure': 0, 'n_div': n_div, 'nogui': False, 'learning_rate': 20.0,
              'learning_rate_final': 0.1,
              'epochs_per_cycle': 10, 'run_cycles': 10, 'display_multiplier': 10}
    s = UIDisplay(**kwargs)
    loss = s.run()
    print("Final loss:  %.6f" % (loss,)) 



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_vertical()
    # sys.exit()
    parsed, kwargs = get_args()

    if parsed.input_image is None and parsed.model_file is None and parsed.test_image is None:
        raise Exception("Need input image (-i) to start training, or model file (-m) to continue.")
    import pprint
    print_args = kwargs.copy()
    print_args['_image_file'] = parsed.input_image
    print_args['_model_file'] = parsed.model_file
    print_args['_test_image'] = parsed.test_image
    pprint.pprint(print_args)
    
    s = UIDisplay(image_file=parsed.input_image,synth_image_name = parsed.test_image, state_file=parsed.model_file, **kwargs)
    s.run()
