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
from util import make_input_grid, downscale_image
import argparse
from threading import Thread
from circular import CircleLayer
from linear import LineLayer
from normal import NormalLayer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.optimizers.schedules import ExponentialDecay

DIV_TYPES = {'circular': CircleLayer, 'linear': LineLayer, 'relu': NormalLayer}


class ScaleInvariantImage(object):
    """
    Train a feed-forward neural network to approximate an image:
    learn f(x, y) = (r,g,b) using every pixel as a training example

    """

    def __init__(self, image_raw, n_hidden, n_dividers, div_type, state_file=None, batch_size=16, learning_rate=.1, sharpness=1000.0, grad_sharpness=3.0):
        """
        :param image_raw: a HxWx3 or HxW numpy array containing the target image.  Training will be wrt downsampled versions of this image.
        :param n_hidden: number of hidden units in the middle
        :param n_dividers: number of input units
        :param div_type: type of division layer, one of DIV_TYPES
        :param image: if not None, a HxWx3 or HxW numpy array containing the target image
        :param batch_size: training batch size
        :param learning_rate: learning rate for the optimizer
        :param sharpness: sharpness constant for activation function, e.g. f(x) = tanh(x*sharpness) for linear

        """
        self.image_raw = image_raw
        self.n_hidden = n_hidden
        self.n_dividers = n_dividers
        self.div_type = div_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.grad_sharpness = grad_sharpness
        self.image_train = None
        self._downscale_level = None
        self.sharpness = sharpness
        self.cycle = 0  # increment for each call to train_more()
        
        if state_file is not None:
            # These params can't change (except for updating weights in self._model), so they override the args.
            # (so they can be None in the args)
            state = ScaleInvariantImage._load_state(state_file)
            weights = state['weights']
            self.cycle, self.image_raw, self.n_hidden, self.n_dividers, self.div_type, self.sharpness, self.grad_sharpness = \
                  state['cycle'], state['image_raw'], state['n_hidden'], state['n_dividers'], state['div_type'], state['sharpness'], state['grad_sharpness']
            # Checks
            if image_raw is not None and image_raw.shape != self.image_raw.shape:
                logging.warning("Image shape doesn't match loaded state:  %s vs %s, using CMD-line argument (will save with this one too)" %
                                (image_raw.shape, self.image_raw.shape))
                self.image_raw = image_raw
        else:
            weights = None

        self._model = self._init_model()
        if weights is not None:
            logging.info("Restored model weights from file:  %s  (resuming at cycle %i)" % (state_file, self.cycle))
            self._model.set_weights(weights)
        else:
            logging.info("Initialized new model.")

        self._model.compile(loss='mean_squared_error',
                            optimizer=tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate))

        logging.info("Model compiled with learning_rate:  %f" % (self.learning_rate,))

    def _make_train(self, downscale, keep_aspect=True):
        
        self._downscale_level = downscale
        self.image_train = downscale_image(self.image_raw, self._downscale_level)

        in_x, in_y = make_input_grid(self.image_train.shape, keep_aspect=keep_aspect)
            
        grid_shape = in_x.shape
        input = np.hstack((in_x.reshape(-1, 1), in_y.reshape(-1, 1)))
        r, g, b = cv2.split(self.image_train / 255.0)
        output = np.hstack((r.reshape(-1, 1), g.reshape(-1, 1), b.reshape(-1, 1)))
        logging.info("Made inputs %s spanning [%.3f, %.3f] and [%.3f, %.3f], %i samples total. <------------" %
                     (grid_shape, input[:, 0].min(), input[:, 0].max(), input[:, 1].min(), input[:, 1].max(), input.shape[0]))
        return input, output

    def _init_model(self):
        """
        Initialize the TF model, either from scratch
        :param n_hidden: number of hidden units in the middle
        :param n_dividers: number of input units
        """

        # Input is just the (x,y) pixel coordinates (scaled)
        input = Input((2,))

        # Use just a line layer, and some ReLu units to color the regions partitioned by the lines
        if self.div_type in DIV_TYPES:
            layer_class = DIV_TYPES[self.div_type]
            div_layer = layer_class(self.n_dividers, grad_sharpness=self.grad_sharpness, input_shape=(
                2,), sharpness=self.sharpness)(input) if self.div_type != 'relu' else layer_class(self.n_dividers, input_shape=(2,))(input)
        else:
            raise Exception("Unknown division type:  %s, must be one of:  %s." % (self.div_type,
                                                                                  ', '.join(DIV_TYPES.keys())))

        middle = Dense(self.n_hidden, activation=tf.nn.relu, use_bias=True,
                       kernel_initializer='random_normal')(div_layer)
        output = Dense(3, use_bias=True, activation=tf.nn.sigmoid)(middle)
        model = Model(inputs=input, outputs=output)
        return model

    def train_more(self, epochs, downscale, verbose=True):

        input, output = self._make_train(downscale)

        # Save numpy training set:
        np.savez_compressed("training_data.npz", input=input, output=output, img_shape=self.image_train.shape)
        print("SAVED Numpy TEST data to file: training_data.npz")

        rand = np.random.permutation(input.shape[0])
        input = input[rand]
        output = output[rand]
        batch_losses = BatchLossCallback()
        self._model.fit(input, output, epochs=epochs,
                        batch_size=self.batch_size, verbose=verbose, callbacks=[batch_losses])
        # Get loss for each step
        batch_history = batch_losses.batch_losses
        logging.info("Batch losses for cycle %i: %s steps" % (self.cycle, len(batch_history)))
        self.cycle += 1
        return batch_history

    def gen_image(self, output_shape, border=0.0, keep_aspect=True):

        x, y = make_input_grid(output_shape, resolution=1.0, border=border, keep_aspect=keep_aspect)
        shape = x.shape
        logging.info("Making display image with shape:  %s" % (shape,))
        inputs = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        logging.info("Rescaled inputs to span [%.3f, %.3f] and [%.3f, %.3f]." % (np.min(inputs[:, 0]),
                                                                                 np.max(inputs[:, 0]),
                                                                                 np.min(inputs[:, 1]),
                                                                                 np.max(inputs[:, 1])))
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
        print("Saving model weights with %i layers to:  %s" % (len(weights), model_filename))
        data = {'weights': weights,
                'image_raw': self.image_raw,
                'cycle': self.cycle,
                'n_dividers': self.n_dividers,
                'n_hidden': self.n_hidden,
                'div_type': self.div_type,
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
        :return: the model, image_raw, n_hidden, n_dividers, div_type
        """
        if not os.path.exists(model_filepath):
            raise Exception("State file doesn't exist:  %s" % (model_filepath,))
        with open(model_filepath, 'rb') as infile:
            state = cp.load(infile)
        print("Loaded model state from:  %s, n_weight_layers: %i" % (model_filepath, len(state['weights'])))

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

    def __init__(self, state_file=None, image_file=None, just_image=False, border=0.0, div_type='linear', frame_dir=None, max_cycles=0,
                 epochs_per_cycle=1, display_multiplier=1.0, downscale=1.0,  n_dividers=40, n_hidden=40, learning_rate=0.001, **kwargs):
        self._border = border
        self._epochs_per_cycle = epochs_per_cycle
        self._display_multiplier = display_multiplier
        self.n_dividers = n_dividers
        self.n_hidden = n_hidden
        self.max_cycles = max_cycles
        self.div_type = div_type
        self._shutdown = False
        self._frame_dir = frame_dir
        self._cycle = 0  # epochs_per_cycle epochs of training and an output update increments this
        self._annotate = False
        self._learning_rate = learning_rate
        self._downscale = downscale  # constant downscale factor for training image
        self._learning_rate_init = learning_rate
        self._output_image = None
        self._loss_history = []

        if not os.path.exists(image_file):
            raise Exception("Image file doesn't exist:  %s" % (image_file,))

        self._image_raw = cv2.imread(image_file)[:, :, ::-1]

        logging.info("Loaded image %s with shape %s." %
                        (image_file, self._image_raw.shape))
        # model/image file prefix is the image bare name without extension
        self._file_prefix = os.path.basename(os.path.splitext(os.path.basename(image_file))[0])

        self._sim = ScaleInvariantImage(n_dividers=n_dividers, n_hidden=n_hidden, learning_rate=self._learning_rate,
                                        div_type=self.div_type, state_file=state_file, image_raw=self._image_raw, **kwargs)

        logging.info("Using output file prefix:  %s" % (self._file_prefix,))

        self._output_shape = None  # for saving:  set during training, train_image.shape * display_multiplier, high res-generated image
        self._frame = None

        self._interactive = not just_image
        if just_image:
            self._write_image()
            self._shutdown = True
            return
        
    def get_filename(self, which='model'):
        if which == 'model':
            return "%s_model_%s_%id_%ih.pkl" % (self._file_prefix, self._sim.div_type, self._sim.n_dividers, self._sim.n_hidden)
        elif which == 'image':
            return "%s_%s_%id_%ih_cycle-%.8i.png" % (self._file_prefix, self._sim.div_type, self._sim.n_dividers, self._sim.n_hidden, self._sim.cycle)
        else:
            raise Exception("Unknown filename type:  %s" % (which,))

    def _train_thread(self):
        # TF 2.x doesn't use sessions or graphs in eager execution
        logging.info("Starting training:")
        logging.info("\tBatch size: %i" % (self._sim.batch_size,))
        logging.info("\tDiv type:  %s" % (self.div_type,))
        logging.info("\tDivider units:  %i" % (self.n_dividers,))
        logging.info("\tHidden units:  %i" % (self.n_hidden,))
        logging.info("\tSharpness: %f" % (self._sim.sharpness,))
        logging.info("\tGradient sharpness: %f" % (self._sim.grad_sharpness,))

        self._cycle = self._sim.cycle  # in case resuming from a saved state

        while not self._shutdown:
            if self._shutdown:
                break

            logging.info("Training batch_size: %i, cycle: %i, downscale_level: %.2f" %
                         (self._sim.batch_size, self._sim.cycle, self._downscale))

            new_losses = self._sim.train_more(self._epochs_per_cycle, downscale=self._downscale)
 
            self._loss_history.append(new_losses)

            # self._save_state()
            if self._shutdown:
                break

            self._output_image = self._write_image() # Create & save image

            filename = self.get_filename('model')
            out_path = filename # Just write to cwd instead of os.path.join(img_dir, filename)
            self._sim.save_state(out_path)  # TODO:  Move after each cycle
            logging.info("Saved model state to:  %s" % (out_path,))

            self._cycle += 1

            if self.max_cycles > 0 and self._cycle >= self.max_cycles:
                logging.info("Reached max cycles (%i), stopping." % (self.max_cycles,))
                self._shutdown = True
                break

 
        # print FFMPEG command to make a movie from the images
        if self._frame_dir is not None:
            logging.info("To make a movie from the images, try:")
            logging.info("  ffmpeg -y -framerate 10 -i %s_%s_%id_%ih_cycle-%%08d.png -c:v libx264 -pix_fmt yuv420p output_movie.mp4" %
                         (os.path.join(self._frame_dir, self._file_prefix), self.div_type, self.n_dividers, self.n_hidden))

    def _write_image(self):
        self._out_shape = (np.array(self._sim.image_train.shape[:2]) * self._display_multiplier).astype(int)
        img = self._sim.gen_image(output_shape=self._out_shape, border=self._border)
        if self._frame_dir is not None:
            out_filename = self.get_filename('image')
            out_path = os.path.join(self._frame_dir, out_filename)
            if os.path.exists(out_path):
                logging.warning("Output image %s already exists, overwriting!!!!" % (out_path,))
            cv2.imwrite(out_path, np.uint8(255 * img[:, :, ::-1]))
            logging.info("Wrote:  %s" % (out_path,))
        return img

    def _start(self):
        #self._train_thread()
        self._worker = Thread(target=self._train_thread)
        self._worker.start()
        logging.info("Started worker thread.")

    def _keypress_callback(self, event):
        if event.key == 'x' or event.key == 'q' or event.key == 'escape':
            logging.info("Shutdown requested, waiting for worker to finish...")
            self._shutdown = True
        else:
            logging.info("Unassigned key: %s" % (event.key,))

    def run(self):
        """
        Run as interactive matplotlib window, updating when new image is available.
           * left is the current training image
           * middle is the current output image
           * right is the loss history
        """
        
        self._start()

        while self._sim.image_train is None:
            time.sleep(0.05)

        plt.ion()
        fig = plt.figure(figsize=(10, 5))
        if self._image_raw.shape[0] > self._image_raw.shape[1]:
            grid = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[3, 3, 2])
            train_ax = fig.add_subplot(grid[0, 0])
            out_ax = fig.add_subplot(grid[0, 1])
            loss_ax = fig.add_subplot(grid[0, 2])
        else:
            grid = gridspec.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,1])
            train_ax = fig.add_subplot(grid[0, 0])
            out_ax = fig.add_subplot(grid[0, 1])
            loss_ax = fig.add_subplot(grid[1, :])

        fig.canvas.mpl_connect('key_press_event', self._keypress_callback)

        artists = {'loss': None, 'train_img': None, 'out_img': None}

        # start main UI loop (training is in the thread):
        while not self._shutdown:
            if artists['train_img'] is None:
                artists['train_img'] = train_ax.imshow(self._sim.image_train)
                train_ax.set_title("Training Image, Cycle %i, downscale:  %.2f\n, image_shape: %s" %
                                   (self._cycle, self._downscale, self._sim.image_train.shape))
                train_ax.axis("off")
            else:
                artists['train_img'].set_data(self._sim.image_train)
                train_ax.set_title("Training Image, Cycle %i, downscale:  %.2f\n, image_shape: %s" %
                                   (self._cycle, self._downscale, self._sim.image_train.shape))
                train_ax.axis("off")

            if self._output_image is not None:
                # Can take a while to generate the first image
                if artists['out_img'] is None:
                    artists['out_img'] = out_ax.imshow(self._output_image)
                    out_ax.set_title("Output Image, shape: %s" % (self._output_image.shape,))
                    out_ax.axis("off")
                else:
                    artists['out_img'].set_data(self._output_image)
                    out_ax.set_title("Output Image, shape: %s" % (self._output_image.shape,))
                    out_ax.axis("off")
                    out_ax.set_aspect('equal')

            if len(self._loss_history) > 0:
                """
                Plot the individual minibatch losses, and their mean per cycle on top.   
                """
                all_losses = np.hstack(self._loss_history)
                # print(self._cycle,"<-----------------------")
                cycle_x = np.linspace(0, self._cycle+1, all_losses.size)
                total_xy = np.array((cycle_x, all_losses)).T
                if artists['loss'] is None:
                    artists['loss'] = loss_ax.plot(total_xy[:, 0], total_xy[:, 1], 'b.', label='Loss per step')[0]
                    loss_ax.set_xlabel("Cycle (%i epochs)" % (self._epochs_per_cycle,))
                    loss_ax.set_ylabel("Loss")
                    loss_ax.legend()
                    loss_ax.set_title("Training Loss History")
                else:
                    artists['loss'].set_data(total_xy[:, 0], total_xy[:, 1])

                # y-axis log
                loss_ax.set_yscale('log')
                loss_ax.relim()
                loss_ax.autoscale_view()

            fig.tight_layout()
            plt.show()
            plt.pause(0.2)

            if self._shutdown:
                break

def get_args():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Learn an image.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--n_dividers", help="Number of divisions (linear or circular) to use.", type=int, default=1)
    parser.add_argument("-n", "--n_hidden", help="Number of hidden_units units in the model.", type=int, default=64)
    parser.add_argument("-t", "--type", help="Type of division unit ('circular' or 'linear' or 'relu').",type=str, default='linear')
    parser.add_argument("-i", "--input_image", help="image to transform", type=str, default=None)
    parser.add_argument("-e", "--epochs", help="Number of epochs between frames.", type=int, default=1)
    parser.add_argument("-x", "--disp_mult", help="Display image dimension multiplier.", type=float, default=1.0)
    parser.add_argument("-m", "--model_file", help="model to load & continue.", type=str, default=None)
    parser.add_argument("-j", "--just_image", help="Just do an image, no training.", action='store_true', default=False)
    parser.add_argument("-b", "--border", help="Extrapolate outward from the original shape by this factor.",type=float, default=0.0)
    parser.add_argument("-p", "--downscale", help="downscale image by this factor, speeds up training at the cost of detail.", type=float, default=1.0)
    parser.add_argument('-l', "--learning_rate", help="Learning rate for the optimizer.", type=float, default=.01)
    parser.add_argument('-s', "--sharpness", help="Sharpness constant for activation function, "+
                        "activation(excitation) = tanh(excitation*sharpness).", type=float, default=1000.0)
    parser.add_argument("-g", '--gradient_sharpness', help="Use false gradient with this sharpness (tanh(grad_sharp * cos_theta)) instead of actual" +
                                                           " (very flat) gradient.  High values result in divider units not moving very much, too low" +
                                                           " and they don't settle.", type=float, default=5.0)
    parser.add_argument('-f', '--save_frames', help="Save frames during training to this directory (must exist).", type=str, default=None)
    parser.add_argument('-c', '--cycles', help="Number of training cycles to do (epochs_per_cycle epochs each). 0 = run forever.", type=int, default=0)
    parsed = parser.parse_args()

    kwargs = {'epochs_per_cycle': parsed.epochs, 'display_multiplier': parsed.disp_mult,
              'border': parsed.border, 'sharpness': parsed.sharpness, 'grad_sharpness': parsed.gradient_sharpness,
              'downscale': parsed.downscale, 'n_dividers': parsed.n_dividers, 'frame_dir': parsed.save_frames,
              'just_image': parsed.just_image, 'n_hidden': parsed.n_hidden, 'max_cycles': parsed.cycles,
              'div_type': parsed.type, 'learning_rate': parsed.learning_rate}
    return parsed, kwargs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parsed, kwargs = get_args()
    
    if parsed.input_image is None and parsed.model_file is None:
        raise Exception("Need input image (-i) or model file (-m).")

    s = UIDisplay(image_file=parsed.input_image,state_file=parsed.model_file ,**kwargs)
    s.run()
