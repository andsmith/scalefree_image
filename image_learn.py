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
from util import make_input_grid, make_central_weights, sample_image,interpolate_weights
import argparse
from threading import Thread
from circular import CircleLayer
from linear import LineLayer
from normal import NormalLayer
import matplotlib.pyplot as plt
import json
from skimage import measure
import matplotlib.gridspec as gridspec
from copy import deepcopy


DIV_TYPES = {'circular': CircleLayer, 'linear': LineLayer, 'sigmoid': NormalLayer}


class ScaleInvariantImage(object):
    """
    Train a feed-forward neural network to approximate an image:
    learn f(x, y) = (r,g,b) using every pixel as a training example

    """

    def __init__(self, image, n_hidden, n_div, state_file=None, batch_size=64, sharpness=1000.0, grad_sharpness=3.0, 
                 learning_rate_initial=0.1, train_size=8192, center_weight_params=None, **kwargs):
        """
        :param image: a HxWx3 or HxW numpy array containing the target image.  Training will be sampled to fit the size.
        :param n_hidden: number of hidden units in the middle
        :param n_div: dictionary containing the number of input units for each division type
        :param n_div_s: number of input units for sigmoid division
        :param state_file: if not None, a file to load the model state from (overrides other args except learning rate, batch size)
        :param image: if not None, a HxWx3 or HxW numpy array containing the target image
        :param batch_size: training batch size
        :param sharpness: sharpness constant for activation function, e.g. f(x) = tanh(x*sharpness) for linear
        :param grad_sharpness: sharpness constant for gradient of activation function, e.g. f'(x) = sharpness * sech^2(x*sharpness)
        :param learning_rate_initial: initial learning rate for Adadelta optimizer
        :param train_size: size of the training set (number of pixels to sample for training)
        :param center_weight_params: if not None, a dict with keys 'weight' (float) and 'sigma' (float) to weight center pixels more heavily

        """
        self.image = image
        self._weight_grid = None
        self.n_hidden = n_hidden
        self.n_div = n_div
        self.batch_size = batch_size
        self.grad_sharpness = grad_sharpness
        self._sample_weights = None
        self.sharpness = sharpness
        self.cycle = 0  # increment for each call to train_more()
        self._center_weight_params = center_weight_params
        self._learning_rate = learning_rate_initial
        self.last_loss = -1
        self.train_size = train_size


        if state_file is not None:
            # These params can't change (except for updating weights in self._model), so they override the args.
            # (so they can be None in the args)
            state = ScaleInvariantImage._load_state(state_file)
            weights = state['weights']
            self.cycle, self.image, self.n_hidden, self.n_div,  self.sharpness, self.grad_sharpness, self.train_size = \
                state['cycle'], state['image'], state['n_hidden'], state['n_div'], state['sharpness'], \
                    state['grad_sharpness'], state['train_size']
            # Checks
            if image is not None and image.shape != self.image.shape:
                logging.warning("Image shape doesn't match loaded state:  %s vs %s, using CMD-line argument (will save with this one too)" %
                                (image.shape, self.image.shape))
                self.image = image

            if self.train_size != train_size:
                logging.info("Warning: Command-line is overriding model's train size:  was %.3f, using %.3f" %
                                (self.train_size, train_size))
                self.train_size = train_size
        else:
            weights = None
            
        # Cache this
        self._last_train_size = None


        if self._center_weight_params is not None:
            train_img_size_wh = self.image.shape[1], self.image.shape[0]
        
            self._weight_grid = make_central_weights(train_img_size_wh, 
                                                    max_weight=self._center_weight_params['weight'],
                                                    rad_rel=self._center_weight_params['sigma'], 
                                                    flatness=self._center_weight_params['flatness'],
                                                    offsets_rel=self._center_weight_params['xy_offsets_rel'])
            logging.info("Using center-weighted samples with max weight %.1f and sigma %.3f (image shape: %s)" %
                        (self._center_weight_params['weight'], self._center_weight_params['sigma'], train_img_size_wh))
            self.weight_cross_sections = {'x': self._weight_grid[self._weight_grid.shape[0]//2,:],
                                        'y': self._weight_grid[:,self._weight_grid.shape[1]//2]}
        else:
            logging.info("Using uniform sample weights.")
            self._weight_grid = None
            self.weight_cross_sections = None

        # Restore trained weights
        self._model = self._init_model()
        if weights is not None:
            logging.info("Restored model weights from file:  %s  (resuming at cycle %i)" % (state_file, self.cycle))
            self._model.set_weights(weights)
        else:
            logging.info("Initialized new model.")

        self._model.compile(loss='mean_squared_error',
                            optimizer=tf.keras.optimizers.Adadelta(learning_rate=self._learning_rate))  # default 0.001

        logging.info("Model compiled with default learning_rate:  %f" % (self._learning_rate,))

    def _make_train(self, train_size):
        input, output, weights = sample_image(self.image, train_size, weights = self._weight_grid)
        
        logging.info("Made inputs %s spanning [%.3f, %.3f] and [%.3f, %.3f], %i samples total." %
                     (input.shape[0], input[:, 0].min(), input[:, 0].max(), input[:, 1].min(), input[:, 1].max(), input.shape[0]))
        return input, output, weights

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

        # Input is just the (x,y) pixel coordinates (scaled)
        input = Input((2,))

        div_layers = []
        for div_type, n_div in self.n_div.items():
            if n_div > 0:
                layer = DIV_TYPES[div_type](n_div, sharpness=self.sharpness,
                                            grad_sharpness=self.grad_sharpness, name="%s_div_layer" % (div_type,))(input)
                div_layers.append(layer)
        if len(div_layers) == 0:
            raise Exception("Need at least one division unit (circular, linear, or sigmoid).")
        if len(div_layers) > 1:
            concat_layer = tf.keras.layers.Concatenate()(div_layers)
        else:
            concat_layer = div_layers[0]

        color_layer = Dense(self.n_hidden, activation=tf.nn.relu, use_bias=True,
                            kernel_initializer='random_normal')(concat_layer)
        output = Dense(3, use_bias=True, activation=tf.nn.sigmoid)(color_layer)
        model = Model(inputs=input, outputs=output)
        return model

    def _update_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
        self._model.optimizer.learning_rate.assign(self._learning_rate)
        logging.info("Updated learning rate to:  %.6f" % (self._learning_rate,))

    def train_more(self, epochs, learning_rate=None, verbose=True):
        if learning_rate is not None and learning_rate != self._learning_rate:
            self._update_learning_rate(learning_rate)

        input, output, weights = self._make_train(self.train_size)

        # Save numpy training set:
        # np.savez_compressed("training_data.npz", input=input, output=output, img_shape=self.image_train.shape)

        rand = np.random.permutation(input.shape[0])
        input = input[rand]
        output = output[rand]
        weights = weights[rand]
        # Apply the same permutation to sample weights to keep them aligned with input/output
        batch_losses = BatchLossCallback()
        swt = ", sample weight range [%.6f, %.6f]" % (np.min(weights), np.max(weights)) if weights is not None else ""
        logging.info("... More training with %i epochs%s" % (epochs, swt))
        self._model.fit(input, output, epochs=epochs, sample_weight=weights,
                        batch_size=self.batch_size, verbose=verbose, callbacks=[batch_losses])
        # Get loss for each step
        batch_history = batch_losses.batch_losses
        logging.info("Batch losses for cycle %i: %s steps" % (self.cycle, len(batch_history)))
        self.cycle += 1
        last_epoch_loss = np.mean(batch_history[-1])
        self.last_loss = last_epoch_loss
        return batch_history, last_epoch_loss

    def gen_image(self, output_shape, border=0.0, keep_aspect=True):

        x, y = make_input_grid(output_shape, resolution=1.0, border=border, keep_aspect=keep_aspect)
        shape = x.shape
        logging.info("Making display image with shape:  %s" % (shape,))
        inputs = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        logging.info("Rescaled inputs to span [%.3f, %.3f] and [%.3f, %.3f], %i total samples." % (np.min(inputs[:, 0]),
                                                                                                   np.max(inputs[:, 0]),
                                                                                                   np.min(inputs[:, 1]),
                                                                                                   np.max(inputs[:, 1]),
                                                                                                   inputs.shape[0]))
        rgb = self._model.predict(inputs)
        logging.info("Display spans:  [%.3f, %.3f]" % (np.min(rgb), np.max(rgb)))
        img = cv2.merge((rgb[:, 0].reshape(shape[:2]), rgb[:, 1].reshape(shape[:2]), rgb[:, 2].reshape(shape[:2])))
        n_clipped = np.sum(img < 0) + np.sum(img > 1)
        logging.info("Display clipping:  %i (%.3f %%)" % (n_clipped, float(n_clipped)/img.size * 100.0))
        img[img < 0.] = 0.
        img[img > 1.] = 1.
        return img

    def save_state(self, model_filename):
        """
        Save only the necessary args in a dict 
        """
        weights = self._model.get_weights()
        logging.info("Saving model weights with %i layers to:  %s" % (len(weights), model_filename))
        data = {'weights': weights,
                'image': self.image,
                'train_size': self.train_size,
                'cycle': self.cycle,
                'n_div': self.n_div,
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
        self.batch_losses = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        # Optional: print the loss for each batch
        # print(f"Batch {batch+1} loss: {logs['loss']:.4f}")


class UIDisplay(object):

    """
    Show image evolving as the model trains.
    After epochs_per_cycle epochs, save the model and write an output image, update display.

    If resuming from a saved state, continue the run number from the saved state.


    """
    
    _CUSTOM_LAYERS = {'CircleLayer': CircleLayer, 'LineLayer': LineLayer, 'NormalLayer': NormalLayer}
    # Variables possible for kwargs, use these defaults if missing from kwargs

    def __init__(self, state_file=None, image_file=None, just_image=False, border=0.0, frame_dir=None, run_cycles=0,batch_size=32,center_weight_params=None,
                 epochs_per_cycle=1, display_multiplier=1.0, train_size=8192,  n_div={}, n_hidden=40, learning_rate=0.1, learning_rate_final=None,
                 nogui=False, nooutput=False, **kwargs):
        self._border = border
        self._epochs_per_cycle = epochs_per_cycle
        self._display_multiplier = display_multiplier
        self.n_div = n_div
        self.n_hidden = n_hidden
        self._center_weight_params = center_weight_params
        self._update_plots = False
        self._run_cycles = run_cycles
        self._shutdown = False
        self._frame_dir = frame_dir
        self._cycle = 0  # epochs_per_cycle epochs of training and an output update increments this
        self._annotate = False
        self._learn_rate = learning_rate  # updates as we anneal
        self._train_size = train_size 
        self._learning_rate_init = learning_rate
        self._learning_rate_final = learning_rate_final
        self._output_image = None
        self._loss_history = []
        self._l_rate_history = []
        self._batch_size = batch_size
        self._nogui = nogui
        self._nooutput = nooutput
        self._metadata = []
        if learning_rate_final is not None:
            if learning_rate_final < 0:
                raise Exception("Final learning rate must be non-negative for annealing.")
            if self._run_cycles <= 0:
                raise Exception("Must specify run_cycles > 0 if annealing with learning_rate_final.")
            if learning_rate >0:
                self._learning_rate_decay = (learning_rate_final / learning_rate) ** (1.0 / (self._run_cycles-1)) if self._run_cycles > 1 else 1.0
                logging.info("Using learning rate annealing:  initial: %.6f, final: %.6f, decay: %.6f per cycle over %i cycles." %
                            (learning_rate, learning_rate_final, self._learning_rate_decay, self._run_cycles))
            else:
                self._learning_rate_decay = 0.0
        else:
            self._learning_rate_decay = None

        if not os.path.exists(image_file):
            raise Exception("Image file doesn't exist:  %s" % (image_file,))

        self._image_raw = cv2.imread(image_file)[:, :, ::-1]

        logging.info("Loaded image %s with shape %s." %
                     (image_file, self._image_raw.shape))
        # model/image file prefix is the image bare name without extension
        self._file_prefix = os.path.basename(os.path.splitext(os.path.basename(image_file))[0])

        self._sim = ScaleInvariantImage(n_div=self.n_div, n_hidden=n_hidden, learning_rate_initial=self._learn_rate,
                                        batch_size=self._batch_size, state_file=state_file, image=self._image_raw, 
                                        train_size=self._train_size, center_weight_params=self._center_weight_params, **kwargs)

        # Check for metadata file
        if state_file is not None:
            meta_filename = self.get_filename('metadata')
            if os.path.exists(meta_filename):
                with open(meta_filename, 'r') as f:
                    metadata = json.load(f)
                logging.info("Loaded metadata from:  %s" % (meta_filename,))
                self._metadata = metadata['frames']  if 'frames' in metadata else metadata
                if 'loss_history' in metadata:
                    self._loss_history = metadata['loss_history'] 
                else:
                    logging.warning("Metadata found but contains no loss history")
                    self._loss_history = []
                if 'learning_rate_history' in metadata:
                    self._l_rate_history = metadata['learning_rate_history'] if 'learning_rate_history' in metadata else []
                else:
                    logging.warning("Metadata found but contains no learning rate history")
                    self._l_rate_history = []
            else:
                logging.info("No metadata file found:  %s" % (meta_filename,))

        logging.info("Using output file prefix:  %s" % (self._file_prefix,))

        self._output_shape = None  # for saving:  set during training, train_image.shape * display_multiplier, high res-generated image
        self._frame = None

        self._just_image = just_image

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
        arch_str += "_%ih" % (self.n_hidden,)
        return arch_str

    def get_filename(self, which='model'):
        if which == 'model':
            return "%s_model_%s.pkl" % (self._file_prefix, self._get_arch_str())
        elif which == 'frame':
            return "%s_output_%s_cycle-%.8i.png" % (self._file_prefix, self._get_arch_str(), self._sim.cycle)
        elif which == 'single-image':
            return "%s_single_%s.png" % (self._file_prefix, self._get_arch_str())
        elif which == 'train-image':
            return "%s_train_%s_tsize=%i.png" % (self._file_prefix, self._get_arch_str(), self._train_size)
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

    def _train_thread(self, max_iter=0):
        # TF 2.x doesn't use sessions or graphs in eager

        # Print full model architecture
        self._sim._model.summary()

        logging.info("Starting training:")
        logging.info("\tBatch size: %i" % (self._sim.batch_size,))
        logging.info("\tDiv types:  %s" % (self._get_arch_str(),))
        logging.info("\tHidden units:  %i" % (self.n_hidden,))
        logging.info("\tSharpness: %f" % (self._sim.sharpness,))
        logging.info("\tGradient sharpness: %f" % (self._sim.grad_sharpness,))

        self._cycle = self._sim.cycle  # in case resuming from a saved statese\\
        run_cycle = 0  # number of cycles since starting this run (not counting previous runs if resuming from a saved state)

        if self._run_cycles > 0:
            logging.info("Will run for %i cycles of %i epochs each (total %i epochs), starting from cycle %i." %
                         (self._run_cycles, self._epochs_per_cycle, self._run_cycles * self._epochs_per_cycle, self._cycle))

        # write frame before any training (may overwrite last frame if continuing a run).
        if not self._nooutput:
            self._output_image = self._gen_image()  
            logging.info("Generated output image of shape:  %s" % (self._output_image.shape,))       
        else:
            self._output_image = np.zeros((10,10,3), dtype=np.float32)
        #self._save_training_image()
        # if we're continuing, don't store the new output image

        if self._cycle == 0:
            # Initial image, random weights, no training.  (Remove?)
            loss = self._sim.last_loss
            if not self._nooutput:
                frame_name = self._write_frame(self._output_image)  # Create & save image
                init_meta = {'cycle': self._sim.cycle,
                            'learning_rate': self._learn_rate,
                            'loss': loss,
                            'filename': frame_name}
                self._metadata.append(init_meta)
                self._write_metadata()


        while (max_iter==0 or run_cycle < max_iter) and not self._shutdown and (run_cycle < self._run_cycles or self._run_cycles == 0):
            if self._shutdown:
                break
            logging.info("Training batch_size: %i, cycle: %i of %i, learning_rate: %.6f" %
                         (self._sim.batch_size, self._sim.cycle, self._run_cycles, self._learn_rate))

            new_losses, loss = self._sim.train_more(self._epochs_per_cycle, learning_rate=self._learn_rate)
            self._l_rate_history.append(self._learn_rate)

            self._loss_history.append(new_losses)

            # self._save_state()
            if self._shutdown:
                break
            if not self._nooutput:
                self._output_image = self._gen_image()
                frame_name = self._write_frame(self._output_image)  # Create & save image
                self._metadata.append({'cycle': self._sim.cycle, 'learning_rate': self._learn_rate, 'loss': loss, 'filename': frame_name})
                self._write_metadata()
            filename = self.get_filename('model')
            out_path = filename  # Just write to cwd instead of os.path.join(img_dir, filename)
            self._sim.save_state(out_path)  # TODO:  Move after each cycle
            logging.info("Saved model state to:  %s" % (out_path,))

            self._cycle += 1
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

    def _gen_image(self, shape=None):
        if shape is None:
            img_shape = self._sim.image.shape
            shape = (np.array(img_shape[:2]) * self._display_multiplier).astype(int)
        img = self._sim.gen_image(output_shape=shape, border=self._border)
        return img

    def _write_image(self, img):
        out_filename = self.get_filename('single-image')
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
                    'train_size': self._train_size,
                    'loss_history': self._loss_history,
                    'learning_rate_history': self._l_rate_history}
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
        elif event.key == 'a':
            self._show_all_hist = not self._show_all_hist
            logging.info("Toggled loss history display mode:  %s" % ("all cycles" if self._show_all_hist else "last 5 cycles only",))   
            self._update_plots = True
        elif event.key == 'w':
            self._show_weight_contours = not self._show_weight_contours
            logging.info("Toggled weight contour display:  %s" % ("on" if self._show_weight_contours else "off",))   
            self._update_plots = True
        else:
            logging.info("Unassigned key: %s" % (event.key,))

    def run(self, debug_nothread=False):
        """
        Run as interactive matplotlib window, updating when new image is available.
           * left is the current training image
           * middle is the current output image
           * right is the loss history
        """
        if self._just_image:
            img_shape = (np.array(self._sim.image.shape[:2]) * self._display_multiplier).astype(int)
            img = self._gen_image(shape=img_shape)
            self._write_image(img)
            return
        
        if debug_nothread:
            logging.info("Debug mode, running training in main thread.")
            self._worker = None
            self._train_thread(max_iter = 1)  # for debugging, don't start thread, just run in here
        else:
            self._start()
            
        max_hist_cycles = 5
        self._show_all_hist = True  # if false only show last max_hist_cycles for plots

        if self._nogui:
            logging.info("No GUI mode, waiting for training to finish...")
            self._worker.join()
            return
 

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
            
            grid = gridspec.GridSpec(8, 3, width_ratios=[1,1,.7])
            logging.info("Tall images, orienting side-by-side.")
            train_ax = fig.add_subplot(grid[:, 0])
            out_ax = fig.add_subplot(grid[:, 1])
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
            grid = gridspec.GridSpec(8, 2, width_ratios=[2,.7])
            train_ax = fig.add_subplot(grid[:4, 0])
            out_ax = fig.add_subplot(grid[4:, 0])
        
        if self._center_weight_params is not None:            
            lrate_ax = fig.add_subplot(grid[:3, -1])
            loss_ax = fig.add_subplot(grid[3:6, -1],sharex=lrate_ax)
            weight_ax = fig.add_subplot(grid[6:, -1])
            weights_plotted=False
        else:
            lrate_ax = fig.add_subplot(grid[:3, -1])
            loss_ax = fig.add_subplot(grid[3:, -1],sharex=lrate_ax)
            weight_ax = None
            weights_plotted=True
            
                


        lrate_ax.grid(which='major', axis='both')
        loss_ax.grid(which='both', axis='both')
        lrate_ax.tick_params(labeltop=False, labelbottom=False)# turn off x tick labels for lrate
        lrate_ax.set_ylabel("Learning Rate")
        train_ax.axis("off")        
        loss_ax.set_xlabel("Cycle (%i epochs)" % (self._epochs_per_cycle,))
        loss_ax.set_ylabel("Loss")
        cmd = "('a' to show last 5 cycles only)" if self._show_all_hist else "('a' to show all cycles)"
        loss_ax.set_title("Training Loss History\n%s\n1 dot = 1 minibatch (%i samples)" % (  cmd, self._batch_size))

        lrate_ax.set_ylabel("Learning Rate")
        lrate_ax.set_title("Learning Rate History\nCurrent rate: %.6f" % (self._learn_rate,))
        lrate_ax.set_yscale('log')

        fig.canvas.mpl_connect('key_press_event', self._keypress_callback)

        artists = {'loss': None, 'train_img': None, 'out_img': None}
        self._update_plots = False # set when user changes something that requires a plot update
        self._show_weight_contours = True
        # start main UI loop (training is in the thread):
        graph_update_cycle = -1  # last cycle we updated the graph, don't change unless it changes
        
        while not self._shutdown and (self._worker is None or self._worker.is_alive()):
            # try:
            if artists['train_img'] is None or self._update_plots:
                # mask out pixels with zero weight in self._sim._weight_grid
                img_out = self._sim.image.copy()
                if self._center_weight_params is not None and self._sim._weight_grid is not None and self._show_weight_contours:
                    # apply contour lines at 20% intervals
                    n_cont = 7
                    contour_levels = [measure.find_contours(self._sim._weight_grid,level = l) for l in np.linspace(0.0, self._center_weight_params['weight'], n_cont, endpoint=True)[1:]]
                    colors = plt.cm.viridis(np.linspace(0,1,len(contour_levels)))[:,:3]
                    for level_ind, contours in enumerate(contour_levels):
                        for contour in contours:
                            contour = np.round(contour).astype(int)
                            img_out[contour[:, 0], contour[:, 1], :] = colors[level_ind]
                    logging.info("Overlayed weight contours on training image.")

                artists['train_img'] = train_ax.imshow(img_out)
            # else:  # FUTURE: replace if training image changes
            #     artists['train_img'].set_data(self._sim.image_train)
            cmd =( "\n('w' to hide weight contours)" if self._show_weight_contours else "\n('w' to show weight contours)")\
                if self._center_weight_params is not None else ""
            train_ax.set_title("Training cycle %i/%s...\nN samples: %i%s" %
                (self._cycle+1, self._run_cycles if self._run_cycles > 0 else '--',
                    self._train_size, cmd))

            if not weights_plotted:
                if self._center_weight_params is not None and self._sim.weight_cross_sections is not None and weight_ax is not None:
                    weight_ax.cla()
                    xc, yc = self._sim.weight_cross_sections['x'], self._sim.weight_cross_sections['y']
                    weight_ax.plot(np.linspace(0, 1.0, xc.size), xc, 'r-', label='X cross-section')
                    weight_ax.plot(np.linspace(0, 1.0, yc.size), yc, 'b-', label='Y cross-section')
                    weight_ax.set_title("Sample Weights Cross-Sections")
                    #weight_ax.set_ylabel("Relative Weight")
                    #weight_ax.set_xlabel("Pixel Index")
                    weight_ax.legend(fontsize=8)
                    # turn off x axis labels, tickes
                    weight_ax.tick_params(labeltop=False, labelbottom=False, labelleft=True, labelright=False,
                                            left=True, right=False, bottom=False, top=False)
                    weight_ax.set_ylim(0.0, self._center_weight_params['weight'] * 1.1)
                    # add grid
                    weight_ax.grid(which='both', axis='y')
                    weights_plotted=True

            if self._output_image is not None:
                # Can take a while to generate the first image
                out_ax.set_title("Cycle %i loss: %.5f\nOutput Image %s" % (self._cycle, self._loss_history[-1][-1] if len(self._loss_history) > 0 else 0.0, self._output_image.shape))

                if artists['out_img'] is None:
                    artists['out_img'] = out_ax.imshow(self._output_image)
                    out_ax.axis("off")
                else:
                    artists['out_img'].set_data(self._output_image)
                    out_ax.axis("off")
                    out_ax.set_aspect('equal')

            if (len(self._loss_history) > 0 and len(self._l_rate_history )> 0 and self._cycle != graph_update_cycle) or self._update_plots:
                """
                Plot the individual minibatch losses, and their mean per cycle on top.   
                """
                if len(self._loss_history) ==0 or len(self._l_rate_history) == 0:
                    self._update_plots = False
                    continue
                graph_update_cycle = self._cycle
                all_losses = np.hstack(self._loss_history)
                loss_sizes = [np.array(lh).size for lh in self._loss_history]
                cycle_x = np.hstack([np.linspace(c, c+1.0, size, endpoint=False) for c, size in zip(range(len(loss_sizes)), loss_sizes)])
                total_xy = np.array((cycle_x, all_losses)).T
                if artists['loss'] is None:
                    artists['loss'] = loss_ax.plot(total_xy[:, 0], total_xy[:, 1], 'b.', label='Loss per step')[0]
                    loss_ax.set_yscale('log')
                else:
                    artists['loss'].set_data(total_xy[:, 0], total_xy[:, 1])
                first_ind = 0 if self._show_all_hist else all_losses.size- np.sum(loss_sizes[-max_hist_cycles:])
                smallest, biggest = all_losses[first_ind:].min(), all_losses[first_ind:].max()
                
                loss_ax.set_ylim(smallest * 0.9, biggest * 1.1)

                # For each learning rate / cycle pair we need to plot a horizontal line segment, so make the list twice as long
                # and plot each y value twice, advancing x by 1 betweeen the first and second copy of each y value.
                lrate_y = np.repeat(np.array(self._l_rate_history), 2)
                lrate_x0 = np.array(range(len(self._l_rate_history)))
                lrate_x = np.zeros(lrate_y.size)
                lrate_x[0::2] = lrate_x0
                lrate_x[1::2] = lrate_x0 + 1.0
                if artists.get('lrate') is None:
                    artists['lrate'] = lrate_ax.plot(lrate_x, lrate_y, 'r-', label='Learning Rate')[0]
                    lrate_ax.set_yscale('log')

                else:
                    artists['lrate'].set_data(lrate_x, lrate_y)
                first_ind = 0 if self._show_all_hist else max(0, len(self._l_rate_history) - max_hist_cycles)
                lrate_ax.set_ylim(min(self._l_rate_history[first_ind:]) * 0.9, max(self._l_rate_history[first_ind:]) * 1.1)
                lrate_ax.set_title("Learning Rate History\nCurrent rate: %.6f" % (self._learn_rate,))

                # set shared x limit
                if  self._show_all_hist:
                    x_max = self._cycle + (self._cycle+1)/20 
                    x_min = -(self._cycle+1)/20
                else:
                    x_max = self._cycle + .1
                    x_min = max(0, self._cycle - max_hist_cycles) - .1
                lrate_ax.set_xlim(x_min, x_max)
                
            self._update_plots = False

            fig.tight_layout()
            plt.draw()
            fig.canvas.flush_events()  # Force canvas to update
            plt.pause(0.2)
                
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


def get_args():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Learn an image.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_image", help="image to transform", type=str, default=None)
    parser.add_argument("-c", "--circles", help="Number of circular division units.", type=int, default=0)
    parser.add_argument("-l", "--lines", help="Number of linear division units.", type=int, default=0)
    parser.add_argument("-s", "--sigmoids", help="Number of sigmoid division units.", type=int, default=0)

    parser.add_argument("-n", "--n_hidden", help="Number of hidden_units units in the model.", type=int, default=64)
    parser.add_argument("-m", "--model_file", help="model to load & continue.", type=str, default=None)
    parser.add_argument("-j", "--just_image", help="Just do an image, no training.", action='store_true', default=False)
    parser.add_argument("-e", "--epochs_per_cycle", help="Number of epochs between frames.", type=int, default=1)
    parser.add_argument(
        '-k', '--cycles', help="Number of training cycles to do (epochs_per_cycle epochs each). 0 = run forever.", type=int, default=0)
    parser.add_argument("-x", "--disp_mult",
                        help="Display image dimension multiplier (X training image size).", type=float, default=1.0)
    parser.add_argument(
        "-b", "--border", help="Extrapolate outward from the original shape by this factor.", type=float, default=0.0)
    parser.add_argument(
        "-t", "--training_size", help="Image will be sampled to (approx) fit this size.", type=float, default=1.0)
    parser.add_argument('-r', "--learning_rate", help="Learning rate for the optimizer.", type=float, default=.01)
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
                        help="Weigh pixels nearer the center higher during training by this radius / spread / flatness and x, y offsets.",
                        nargs=5, type=float, default=[1.0, 2.0])
    parser.add_argument("--nogui", help="No GUI, just run training to completion.", action='store_true', default=False)
    parser.add_argument("--nooutput", help="No output during training.", action='store_true', default=False)
    parser.add_argument('-z', '--batch_size', help="Training batch size.", type=int, default=32)
    parsed = parser.parse_args()
    n_div = {'circular': parsed.circles, 'linear': parsed.lines, 'sigmoid': parsed.sigmoids}
    center_weight = {'weight': parsed.weigh_center[0], 
                     'flatness': parsed.weigh_center[2],
                     "xy_offsets_rel": (parsed.weigh_center[3], parsed.weigh_center[4]),
                     'sigma': parsed.weigh_center[1]} if parsed.weigh_center[0] != 1.0 else None
    # print("CENTER WEIGHT PARAMS:  ", center_weight)
    kwargs = {'epochs_per_cycle': parsed.epochs_per_cycle, 'display_multiplier': parsed.disp_mult, 'center_weight_params': center_weight,
              'border': parsed.border, 'sharpness': parsed.sharpness, 'grad_sharpness': parsed.gradient_sharpness,
              'train_size': parsed.training_size, 'n_div': n_div, 'frame_dir': parsed.save_frames, 'batch_size': parsed.batch_size,
              'just_image': parsed.just_image, 'n_hidden': parsed.n_hidden, 'run_cycles': parsed.cycles, 'nooutput': parsed.nooutput,
              'learning_rate': parsed.learning_rate, 'nogui': parsed.nogui, 'learning_rate_final': parsed.learning_rate_final}
    return parsed, kwargs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parsed, kwargs = get_args()

    if parsed.input_image is None and parsed.model_file is None:
        raise Exception("Need input image (-i) to start training, or model file (-m) to continue.")
    s = UIDisplay(image_file=parsed.input_image, state_file=parsed.model_file, **kwargs)
    s.run()
