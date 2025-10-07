import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as cp
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np
import cv2
import os
import logging
from util import make_input_grid, make_central_weights
from circular import CircleLayer
from linear import LineLayer
from normal import NormalLayer
import matplotlib.pyplot as plt


DIV_TYPES = {'circular': CircleLayer, 'linear': LineLayer, 'sigmoid': NormalLayer}
import tensorflow as tf

class NoisyOptimizer(tf.keras.optimizers.Optimizer):
    """
    Wrapper optimizer: adds Gaussian noise to gradients, then delegates to `base_optimizer`.
    Note: we pass a dummy learning_rate to super().__init__ because the Optimizer base requires it.
    """
    def __init__(self, base_optimizer, name="NoisyOpt", **kwargs):
        # Optimizer base requires a learning_rate argument in many TF versions.
        # Use a harmless dummy value (we delegate updates to `base_optimizer` anyway).
        super().__init__(learning_rate=0.0, name=name, **kwargs)

        self.base = base_optimizer
        # step counter for indexing sigmas (int64 so it matches many TF counters)
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name="noisy_step")
        self.sigmas = None  # will hold a 1-D tf.Tensor of sigma values if set

    def set_sigmas(self, sigmas):
        """Accepts a list/tuple/1D-np-array or 1D-tf.Tensor of stddevs to use per-step.
        If the step index exceeds the length, the final value will be used (or 0 if not provided)."""
        self.sigmas = tf.convert_to_tensor(list(sigmas), dtype=tf.float32)

    def _add_noise_to_grad(self, g, sigma):
        """Add noise while preserving IndexedSlices (for embeddings)."""
        if g is None:
            return None
        if isinstance(g, tf.IndexedSlices):
            vals = g.values + tf.random.normal(tf.shape(g.values), stddev=sigma)
            return tf.IndexedSlices(vals, g.indices, g.dense_shape)
        else:
            return g + tf.random.normal(tf.shape(g), stddev=sigma)
        
            
    def apply_gradients_new(self, grads_and_vars, name=None, **kwargs):
        # Determine noise level
        if self.sigmas is not None and int(self.step) < len(self.sigmas):
            sigma = self.sigmas[self.step]
        else:
            sigma = 0.0

        # Add Gaussian noise
        noisy = [
            (g + tf.random.normal(tf.shape(g), stddev=sigma) if g is not None else None, v)
            for g, v in grads_and_vars
        ]

        # Step bookkeeping
        self.step.assign_add(1)
        self.iterations.assign_add(1)

        # ✅ Don't pass 'name' or extra kwargs to base optimizer
        return self.base.apply_gradients(noisy)



    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        # compute scalar sigma for this step
        if self.sigmas is None:
            sigma = tf.constant(0.0, dtype=tf.float32)
        else:
            # clamp index so we don't go out of bounds; use last sigma if step >= len(sigmas)
            idx = tf.cast(self.step, tf.int32)
            last_idx = tf.maximum(tf.shape(self.sigmas)[0] - 1, 0)
            idx_clamped = tf.minimum(idx, last_idx)
            sigma = self.sigmas[idx_clamped]

        # add noise to each gradient (properly handling None and IndexedSlices)
        noisy = []
        for g, v in grads_and_vars:
            noisy_g = self._add_noise_to_grad(g, sigma) if g is not None else None
            noisy.append((noisy_g, v))

        # increment counters
        # (self.iterations is provided by the Optimizer base class)
        self.step.assign_add(1)
        self.iterations.assign_add(1)

        # delegate the actual update to the wrapped optimizer
        return self.base.apply_gradients(noisy,**kwargs)

    def get_config(self):
        # serialize base optimizer and sigmas (if present)
        config = super().get_config()
        config.update({
            "base_optimizer": tf.keras.optimizers.serialize(self.base),
            "sigmas": None if self.sigmas is None else self.sigmas.numpy().tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_opt = tf.keras.optimizers.deserialize(config.pop("base_optimizer"))
        sigmas = config.pop("sigmas", None)
        obj = cls(base_opt, **config)
        if sigmas is not None:
            obj.set_sigmas(sigmas)
        return obj

    def update_learning_rate(self, learning_rate):
        self.base.learning_rate.assign(learning_rate)
        logging.info("Updated learning rate to:  %.6f" % (self.base.learning_rate,))


class NNetImage(object):
    """
    Train a feed-forward neural network to approximate an image:
    learn f(x, y) = (r,g,b) using every pixel as a training example

    """

    def __init__(self, *args, **kwargs):
        import pprint
        print("\n\n\n***********************")    
        print("SCALEFREE INIT ARGS:")
        pprint.pprint(args)
        print("\nSCALEFREE INIT KWARGS:")
        image = None
        if 'image_raw' in kwargs:
            image = kwargs['image_raw']
            kwargs['image_raw'] = "Image:  %s" % (image.shape,)  # avoid printing large image array
        pprint.pprint(kwargs)
        if image is not None:
            kwargs['image_raw'] = image
        print("***********************\n\n\n")
        
        self._init(*args, **kwargs)
        
    def _init(self, image, n_hidden, n_structure, n_div, state_file=None, batch_size=64, sharpness=1000.0, grad_sharpness=3.0, 
                 learning_rate_initial=1.0, n_train=0, center_weight_params=None, line_params=3, dry_run=False, **kwargs):
        """
        :param image: a HxWx3 or HxW numpy array containing the target image.  Training will be on this image.
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
        :param n_train: number of training samples to use (0 = all pixel xy positions, else sample this many random positions)
        :param center_weight_params: if not None, a dict with keys: 'r_inner' (float), 'r_outer' (float), 'w_max' (float), and 'xy_offset' (tuple of floats)
        :param line_params: 2 or 3, parameterization of line units (2 = angle + offset, 3 = angle + center + offset)
        """
        self.image = image
        self._line_params = line_params
        self.n_hidden = n_hidden
        self.n_div = n_div
        self.n_structure = n_structure
        self.batch_size = batch_size
        self.grad_sharpness = grad_sharpness
        self.dry_run = dry_run
        self._n_train = n_train
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
            state = NNetImage._load_state(state_file)
            weights = state['weights']
            self.cycle, self.image, self.n_hidden, self.n_structure, self.n_div,  self.sharpness, self.grad_sharpness = \
                state['cycle'], state['image'], state['n_hidden'], state['n_structure'], state['n_div'], state['sharpness'], state['grad_sharpness']                
                
            # Checks & overrides between command line args and state file
            if image is not None and (image.shape != self.image.shape or not np.allclose(image, self.image)):
                logging.warning("This appears to be a new image.  Things might get weird...")
                self.image = image
            if grad_sharpness != self.grad_sharpness:
                logging.info("Using New Gradient Sharpness from state file:  %d" % (grad_sharpness,))
                self.grad_sharpness = grad_sharpness
            if self.n_hidden!= n_hidden:
                logging.warning("Number of hidden units in state file (%d) does not match argument (%d), using state file." % (self.n_hidden, n_hidden))
            if self.n_structure != n_structure:
                logging.warning("Number of structure units in state file (%d) does not match argument (%d), using state file." % (self.n_structure, n_structure))
            if self.n_div['linear'] != n_div.get('linear', 0) or self.n_div['circular'] != n_div.get('circular', 0) or self.n_div['sigmoid'] != n_div.get('sigmoid', 0):
                logging.warning("Number each division unit type in state file (%s) does not match argument (%s), using state file." % (self.n_div, n_div))
        else:
                
            weights = None
            
        # Cache this
        self._input, self._output = self._make_train()


        self._model = self._init_model()
        if weights is not None:
            logging.info("Restored model weights from file:  %s  (resuming at cycle %i)" % (state_file, self.cycle))
            self._model.set_weights(weights)
        else:
            logging.info("Initialized new model.")
            
            
            
        base_optimizer = tf.keras.optimizers.Adadelta(
            learning_rate=1.0, use_ema=False, ema_momentum=0.99
        )

        self._optimizer = NoisyOptimizer(base_optimizer)

        self._model.compile(loss='mean_squared_error', optimizer=self._optimizer)
            
            
            
        # print("Initial learning rate:  %.7f" % self._learning_rate)
        # base_optimizer=tf.keras.optimizers.Adadelta(learning_rate=self._learning_rate, use_ema=False, ema_momentum=0.99)
        # optimizer = NoisyOptimizer(base_optimizer)
        # self._model.compile(loss='mean_squared_error', optimizer=optimizer)  # default 0.001

        logging.info("Model compiled with default learning_rate:  %f" % (self._learning_rate,))

    def _make_train(self, keep_aspect=True):

        in_x, in_y = make_input_grid(self.image.shape, keep_aspect=keep_aspect)

        grid_shape = in_x.shape
        input = np.hstack((in_x.reshape(-1, 1), in_y.reshape(-1, 1)))
        r, g, b = cv2.split(self.image / 255.0)
        output = np.hstack((r.reshape(-1, 1), g.reshape(-1, 1), b.reshape(-1, 1)))
        
        if self._center_weight_params is not None:
            train_img_size_wh = self.image.shape[1], self.image.shape[0]
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
        w, h = self.image.shape[1], self.image.shape[0]

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
        w, h = self.image.shape[1], self.image.shape[0]

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

        aspect_ratio = self.image.shape[1] / self.image.shape[0]
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
        if self.dry_run:
            logging.info("Dry run, skipping loss computation")
            return -1.23456
        loss = self._model.evaluate(self._input, self._output, batch_size=8192, verbose=0)
        logging.info("Current loss at cycle %i:  %.6f" % (self.cycle, loss))
        return loss


    def train_more(self, epochs, learning_rate=None, noise_temps=None, verbose=True):
        if learning_rate is not None and learning_rate != self._learning_rate:
            self._optimizer.update_learning_rate(learning_rate)

        input, output = self._input, self._output

        # Save numpy training set:
        # np.savez_compressed("training_data.npz", input=input, output=output, img_shape=self.image.shape)
        self.anneal_temp = noise_temps[0] if noise_temps is not None else 0.0
        if noise_temps is not None:
            # Langevin dynamics noise:
            noise_sds = np.sqrt(2 * learning_rate * np.array(noise_temps))
        else:
            noise_sds = np.array([0.0])

        rand = np.random.permutation(input.shape[0])
        input = input[rand]
        output = output[rand]
        n_train_samples = input.shape[0]
        # Apply the same permutation to sample weights to keep them aligned with input/output
        sample_weights = self._sample_weights[rand] if self._sample_weights is not None else None
        
        batch_losses = BatchLossCallback(dry_run=self.dry_run,
                                         n_epochs=epochs,
                                         noise_sds=noise_sds, 
                                         n_train=n_train_samples,
                                         batch_size=self.batch_size, 
                                         anneal_temps=noise_temps)
        
        swt = ", sample weight range [%.6f, %.6f]" % (np.min(sample_weights), np.max(sample_weights)) if sample_weights is not None else ""
        logging.info("... More training with %i epochs%s" % (epochs, swt))
        
        self._model.optimizer.set_sigmas(noise_sds)
        
        if not self.dry_run:
            self._fit(input, output, epochs=epochs, sample_weight=sample_weights,
                            batch_size=self.batch_size, verbose=verbose, callbacks=[batch_losses])
        # Get loss for each step
        loss_history = batch_losses.losses
        self.cur_loss = np.mean(loss_history[-1]) if len(loss_history) > 0 else -2

        logging.info("Batch losses for cycle %i has %i epochs, each with %i minibatches" % (self.cycle, len(loss_history), len(loss_history[0])))
        self.cycle += 1
        
        return loss_history
    
    
    def _fit(self, *args, **kwargs):
            self._model.fit(*args, **kwargs)

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
                'image': self.image,
                'cycle': self.cycle,
                'n_train': self._n_train,
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

    def __init__(self, dry_run, n_epochs, n_train, batch_size, noise_sds=None, anneal_temps=None):
        """
        Record the losses from each minibatch & epoch for this cycle. (list of lists)
        
        If dry run, make up fake data, appropriate for the number of minibatches/epoch, and number of epochs
        """
        super().__init__()
        self.dry_run = dry_run
        self.n_epochs = n_epochs
        self.noise_sds = noise_sds
        self.n_train = n_train
        self.batch_size = batch_size
        self.anneal_temps = anneal_temps
        
        self.losses = []  # list for each epoch of all minibatch losses
        if dry_run:
            self.losses = self._gen_fake_data()
            
    def _gen_fake_data(self):
        losses = []
        n_batches = int(np.ceil(float(self.n_train) / float(self.batch_size)))
        base_loss = 0.1
        for e in range(self.n_epochs):
            epoch_losses = []
            for b in range(n_batches):
                noise = 0.05 * np.random.randn()
                epoch_losses.append(base_loss + noise)
            losses.append(epoch_losses)
            base_loss *= 0.95  # decay a bit each epoch
        return losses

    def on_epoch_begin(self, epoch, logs=None):
        self.losses.append([])  # new epoch    

    def on_train_batch_end(self, batch, logs=None):
        self.losses[-1].append(logs['loss'])
        


def test_vertical():
    
    # TEST IMAGE:
    # Synthetic
    # lines = {'centers': np.array([[0.0, 0.2], [0.0, -0.2]]),
    #          'angles': np.array([0, 1])}
    # tim = TestImageMaker(image_size_wh=(20,36))
    # test_image = tim.make_image('spec_image', lines = lines, is_color=False)
    # n_div = {'linear': 2, 'circular': 0, 'sigmoid': 0}    
    
    #  python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 255 -n 64 -p 3 -x 6 -r 1  -e 10 -k 1 -z 32768  -w 10.0 .18 .33 .678 .412  --nogui

    barn_weights =  {'w_max': 10, 
                         'r_inner': .18,
                         'r_outer': .33,
                         'offsets_xy_rel': (.678, .412)}

    
    anneal_args = [10.0, 0.001, 1000]

    # The rest of the training parameters:
    kwargs = {'image_raw': cv2.imread(r'input/barn.png'),  
            'n_div': {'linear': 64, 'circular': 0, 'sigmoid': 0}, 
              'n_hidden': 64, 'n_structure': 255,
              'display_multiplier': 6,
              'n_train': 0,
              'center_weight_params': barn_weights,    
              'learning_rate': 1.0, 
              #'learning_rate_final': 0.0001,  # run_cycles must be > 0 for this to be used
              #'anneal_args': anneal_args, 
              'nogui': True,
              'batch_size': 32768,
              'run_cycles': 1, 'epochs_per_cycle': 10}
    
    s = NNetImage(**kwargs)
    
    # DO cycles manually here:
    for cycle_ind in range(kwargs['run_cycles']):
        print("Cycle %i/%i -------------------------" % (cycle_ind+1, kwargs['run_cycles']))
        # linearly anneal noise temperature from 10 to 0.01 over the cycles
        noise_temps = np.linspace(10, .01, kwargs['epochs_per_cycle'])
        # TODO: implement learning rate decay with learning_rate_final argument
        learning_rate = kwargs['learning_rate']
        loss = s.train_more(epochs=kwargs['epochs_per_cycle'], learning_rate=learning_rate, 
                            noise_temps=noise_temps, verbose=True)
        epoch_means = [np.mean(epoch)for epoch in loss]
        print("Last cycle mean loss:  %.6f,  Last epoch mean loss:  %.6f,  Last minibatch loss:" % (
            np.mean(epoch_means), epoch_means[-1]), loss[-1][-1])
    
    img = s.gen_image((200, 360), border=0.1, keep_aspect=True)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vertical()