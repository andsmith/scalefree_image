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
import numpy as np
import cv2
import time
import os
import logging
from util import make_input_grid
import argparse
from threading import Thread
from circular import CircleLayer
from linear import LineLayer
from normal import NormalLayer
from keras.optimizers.schedules import ExponentialDecay

DIV_TYPES = {'circular': CircleLayer, 'linear': LineLayer, 'relu': NormalLayer}


class ScaleInvariantImage(object):
    """
    Train a feed-forward neural network to approximate an image:
    learn f(x, y) = (r,g,b) using every pixel as a training example

    """

    def __init__(self, n_hidden, n_dividers, div_type, image=None, state=None, batch_size=16, learning_rate=.1, epochs_per_cycle=1, sharpness=1000.0, use_false_gradient=False):
        """
        :param n_hidden: number of hidden units in the middle
        :param n_dividers: number of input units
        :param div_type: type of division layer, one of DIV_TYPES
        :param image: if not None, a HxWx3 or HxW numpy array containing the target image
        :param state: if not None, a dict containing a saved model state
        :param batch_size: training batch size
        :param learning_rate: learning rate for the optimizer
        :param epochs_per_cycle: number of epochs to train per cycle
        :param sharpness: sharpness constant for activation function, e.g. f(x) = tanh(x*sharpness) for linear

        """
        if image is None and state is None:
            raise Exception("Need input image, or state to restore.")
        self.n_hidden = n_hidden
        self._epochs_per_cycle = epochs_per_cycle
        self.n_dividers = n_dividers
        self.div_type = div_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_false_gradient = use_false_gradient
        self._image = image
        self.sharpness = sharpness

        if int(image is None) + int(state is None) != 1:
            raise Exception("Need input image, or state to restore.")

        self._input, self._output = self._make_train()

        self._init_model(state)
        
        # Save training data to numPy files for later analysis
        np.savez('training_data.npz', input=self._input, output=self._output, img_shape=self._image.shape)
        logging.info("Saved training data to training_data.npz")



        logging.info("Training set samples: %i" % (self._input.shape[0],))
        logging.info("Made input:  %s" % (self._input.shape,))
        logging.info("Made output:  %s" % (self._output.shape,))
        logging.info("Input spans:  [%f, %f]" % (np.min(self._input), np.max(self._input)))


    def _calc_training_steps_per_cycle(self):
        n_steps_per_epoch = int(np.ceil(float(self._input.shape[0]) / float(self.batch_size)))
        logging.info("Training steps per epoch:  %i" % (n_steps_per_epoch,))
        return n_steps_per_epoch * self._epochs_per_cycle

    def get_image(self):
        return self._image

    def pixel_to_coord(self, x, y):

        # scale to fit in [-1, 1] x [-1, 1]
        x = (x - self._image.shape[1]/2.0) * self._scale
        y = (y - self._image.shape[0]/2.0) * self._scale

        # add polar coords?
        # r = np.sqrt(x*x + y*y)
        # t = np.abs(np.arctan2(y, x))

        return x, y  # , r, t 

    def _make_train(self, shape=None):
        in_x, in_y = make_input_grid(self._image.shape)
        grid_shape = in_x.shape
        input = np.hstack((in_x.reshape(-1, 1), in_y.reshape(-1, 1)))
        r, g, b = cv2.split(self._image / 255.0)
        output = np.hstack((r.reshape(-1, 1), g.reshape(-1, 1), b.reshape(-1, 1)))
        logging.info("Made inputs %s spanning [%.3f, %.3f] and [%.3f, %.3f]. <------------" %
                     (grid_shape, input[:,0].min(), input[:,0].max(), input[:,1].min(), input[:,1].max()))  

        return input, output

    def get_model(self):
        return self._model

    def _init_model(self, state):
        """
        Initialize the TF model, either from scratch or from a saved state
        :param state: if not None, a dict containing a saved model state
        :param n_hidden: number of hidden units in the middle
        :param n_dividers: number of input units
        """

        if state is not None:
            self._model = state['model']
            self._image = state['image']
            logging.info("Model state restored")
            return

        # Input is just the (x,y) pixel coordinates (scaled)
        input = Input((2,))

        # Use just a line layer, and some ReLu units to color the regions partitioned by the lines
        if self.div_type in DIV_TYPES:
            layer_class = DIV_TYPES[self.div_type]
            if self.div_type == 'linear':
                div_layer = layer_class(self.n_dividers, use_false_gradient=self.use_false_gradient, input_shape=(2,))(input)
            else:
                div_layer = layer_class(self.n_dividers, input_shape=(2,), sharpness=self.sharpness)(input) if self.div_type != 'relu' else layer_class(self.n_dividers, input_shape=(2,))(input)
        else:
            raise Exception("Unknown division type:  %s, must be one of:  %s." % (self.div_type,
                                                                                  ', '.join(DIV_TYPES.keys())))

        middle = Dense(self.n_hidden, activation=tf.nn.relu, use_bias=True,
                       kernel_initializer='random_normal')(div_layer)
        output = Dense(3, use_bias=True, activation=tf.nn.sigmoid)(middle)
        self._model = Model(inputs=input, outputs=output)

        self._model.compile(loss='mean_squared_error',
                            optimizer=tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate))

        logging.info("INIT DONE")

    def train_more(self):
        rand = np.random.permutation(self._input.shape[0])
        self._input = self._input[rand]
        self._output = self._output[rand]

        self._model.fit(self._input, self._output, epochs=self._epochs_per_cycle, batch_size=self.batch_size, verbose=True)

    def get_display_image(self, resolution=1.0, margin=1.0):
        x,y = make_input_grid(self._image.shape, resolution, margin)
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


class UIDisplay(object):

    """
    Show image evolving as the model trains.
    After epochs_per_cycle epochs, save the model and write an output image, update display.

    If resuming from a saved state, continue the run number from the saved state.


    """
    _CUSTOM_LAYERS = {'CircleLayer': CircleLayer, 'LineLayer': LineLayer, 'NormalLayer': NormalLayer}
    # Variables possible for kwargs, use these defaults if missing from kwargs

    def __init__(self, state_file=None, image_file=None, just_image=False, border=1.0, div_type='linear', 
                 epochs_per_cycle=1, display_multiplier=1.0, downsample=1.0,  n_dividers=40, n_hidden=40, learning_rate=0.001, use_false_gradient=False, **kwargs):
        self._border = border
        self._epochs_per_cycle = epochs_per_cycle
        self._display_multiplier = display_multiplier
        self.n_dividers =    n_dividers
        self.n_hidden = n_hidden
        self.div_type = div_type
        self._shutdown = False
        self._epoch = 0
        self._annotate=False
        self._cycle = 0  # number of cycles
        self._learning_rate = learning_rate
        self.use_false_gradient = use_false_gradient

        if int(image_file is None) + int(state_file is None) != 1:
            raise Exception("Need input image, or state to restore.")
        # state
        self._tempdir = tempfile.mkdtemp(prefix="imageStretch_")

        if state_file is not None:
            self._out_dir = os.path.dirname(os.path.abspath(state_file))
            self._load_state(state_file)
        else:
            if not os.path.exists(image_file):
                raise Exception("Image file doesn't exist:  %s" % (image_file,))
            
            self._image = self._downsample_image(cv2.imread(image_file)[:, :, ::-1], downsample)
            logging.info("Loaded image %s with shape %s (downsample factor %.2f):  %i training samples" % (image_file, self._image.shape,downsample, self._image.shape[0]*self._image.shape[1]))
            # for the filename suffix, use the image bare name without extension
            self._suffix = os.path.basename(os.path.splitext(os.path.basename(image_file))[0])
            self._out_dir = os.path.dirname(os.path.abspath(image_file))

            print("KWARGS: ", kwargs)
            self._sim = ScaleInvariantImage(n_dividers=n_dividers, n_hidden=n_hidden, learning_rate=self._learning_rate,
                                            image=self._image, div_type=self.div_type, use_false_gradient=self.use_false_gradient, state=None, epochs_per_cycle=self._epochs_per_cycle, 
                                            **kwargs)
            self._cycle = 0

        logging.info("Using output dir:  %s" % (self._out_dir,))
        logging.info("Using output suffix:  %s" % (self._suffix,))

        self._out_shape = (np.array(self._image.shape[:2]) * self._display_multiplier).astype(int)
        self._frame = None

        self._interactive = not just_image
        if just_image:
            self._write_image()
            self._shutdown = True
            return

        self._frame = self._make_frame()

    def _downsample_image(self, image, factor):
        if factor == 1.0:
            return image
        new_shape = (int(image.shape[1] / factor), int(image.shape[0] / factor))
        logging.info("Downsampling image from %s to %s" % (image.shape, new_shape))
        return cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)

    def get_filename(self, which='model'):
        if which == 'model':
            return "%s_model_%s_%id_%ih_cycle-%.8i.keras" % (self._suffix, self._sim.div_type, self._sim.n_dividers, self._sim.n_hidden, self._cycle)
        elif which == 'image':
            return "%s_output_%s_%id_%ih_cycle-%.8i.png" % (self._suffix, self._sim.div_type, self._sim.n_dividers, self._sim.n_hidden, self._cycle)
        else:
            raise Exception("Unknown filename type:  %s" % (which,))

    def _save_state(self):
        logging.info("Saving state...")
        model_filename = self.get_filename('model')
        model_filepath = os.path.join(self._out_dir, model_filename)
        temp_filepath = os.path.join(self._tempdir, model_filename)
        if os.path.exists(model_filepath):
            logging.warning("Model file %s already exists, overwriting!!!!" % (model_filepath,))
        self._sim.get_model().save(temp_filepath)
        with open(temp_filepath, 'rb') as infile:
            model_bin = infile.read()
        model = {'model_bin': model_bin,
                 'epc': self._epochs_per_cycle,
                 'dm': self._display_multiplier,
                 'suffix': self._suffix,
                 'out_dir': self._out_dir,
                 'image': self._image,
                 'border': self._border,
                 'cycle': self._cycle}

        with open(model_filepath, 'wb') as outfile:
            cp.dump(model, outfile)
        logging.info("Wrote:  %s" % (model_filepath,))

    def _load_state(self, model_filepath):
        logging.info("loading state...")
        with open(model_filepath, 'rb') as infile:
            state = cp.load(infile)
        self._epochs_per_cycle = state['epc']
        self._display_multiplier = state['dm']
        self._suffix = state['suffix']
        self._out_dir = state['out_dir']
        self._image = state['image']
        self._cycle = state['cycle']
        self._border = state['border']
        model_filename = "model_%i%s.tf" % (self._cycle, self._suffix)
        model_filepath = os.path.join(self._tempdir, model_filename)
        with open(model_filepath, 'wb') as outfile:
            outfile.write(state['model_bin'])
        state['model'] = tf.keras.models.load_model(model_filepath, custom_objects=self._CUSTOM_LAYERS)
        self._sim = ScaleInvariantImage(state=state)

    def _train_thread(self):
        # TF 2.x doesn't use sessions or graphs in eager execution
        self._epoch = 0
        logging.info("Starting training:")
        logging.info("\tBatch size: %i" % (self._sim.batch_size,))
        logging.info("\tDiv type:  %s" % (self.div_type,))
        logging.info("\tDivider units:  %i" % (self.n_dividers,))
        logging.info("\tHidden units:  %i" % (self.n_hidden,))
        logging.info("\tTraining samples: %i" % (self._sim._input.shape[0],))
        logging.info("\tSharpness: %f" % (self._sim.sharpness,))

        while not self._shutdown:
            logging.info("Training batch_size: %i, cycle: %i" % (self._sim.batch_size, self._cycle))
            if self._shutdown:
                break

            self._sim.train_more()

            #self._save_state()
            if self._shutdown:
                break
            img = self._write_image()
            self._update_image(img)
            self._cycle += 1
            
        self._save_state()
        logging.info("Close detected, shutting down worker thread.")
        logging.info("Deleting work dir:  %s" % (self._tempdir,))
        shutil.rmtree(self._tempdir)

        # print FFMPEG command to make a movie from the images
        logging.info("To make a movie from the images, try:")
        logging.info("  ffmpeg -y -framerate 10 -i %s_output_%s_%id_%ih_cycle-%%08d.png -c:v libx264 -pix_fmt yuv420p output_movie.mp4" %
                     (self._suffix, self.div_type, self.n_dividers, self.n_hidden))

    def _write_image(self):

        img = self._sim.get_display_image( margin=self._border, resolution=self._display_multiplier)

        out_filename = self.get_filename('image')
        out_path = os.path.join(self._out_dir, out_filename)
        if os.path.exists(out_path):
            logging.warning("Output image %s already exists, overwriting!!!!" % (out_path,))
        out_path = os.path.join(self._out_dir, out_path)

        cv2.imwrite(out_path, np.uint8(255 * img[:, :, ::-1]))
        logging.info("Wrote:  %s" % (out_path,))
        return img

    def _update_image(self, image):
        self._image = image
        self._frame = self._make_frame()
        logging.info("Setting %i x %i image" % (image.shape[1], image.shape[0]))

    def _make_frame(self):
        image = cv2.resize(self._image, (self._out_shape[1], self._out_shape[0]), interpolation=cv2.INTER_NEAREST)
        if not self._annotate:
            return image
        lines = ['Architecture:  divider_type:  %s' % (self.div_type,),
                 'n_dividers:  %i' % (self.n_dividers,),
                 'n_hidden:  %i' % (self.n_hidden,),
                 'Cycle (of %i epochs):  %i' % (self._epochs_per_cycle, self._cycle,),]
        y = 20
        x = 10
        font_scale = 0.75
        for line in lines:
            cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            y += int(30 * font_scale)

        return image

    def _start(self):
        self._worker = Thread(target=self._train_thread)
        self._worker.start()
        logging.info("Started worker thread.")

    def run(self):
        win_name = "Scale-free image learning"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 600, 600)
        # cv2.setMouseCallback(win_name, self._on_mouse)
        self._start()

        while not self._shutdown:
            cv2.imshow(win_name, self._frame[:, :, ::-1])
            k = cv2.waitKey(100) & 0xFF
            if k == 27 or k == ord('q'):  # ESC or 'q' to quit
                logging.info("Shutdown requested, waiting for worker to finish...")
                self._shutdown = True
                break
            elif k == ord('a'):  # 'a' to toggle annotation
                self._annotate = not self._annotate
                logging.info("Annotation set to: %s" % (self._annotate,))
                self._frame = self._make_frame()
            elif k != 255:
                logging.info("Unassigned key: %s" % (chr(k) if 32 <= k < 127 else k,))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Learn an image.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--n_dividers",
                        help="Number of divisions (linear or circular) to use.", type=int, default=1)
    parser.add_argument("-n", "--n_hidden", help="Number of hidden_units units in the model.", type=int, default=64)
    parser.add_argument("-t", "--type", help="Type of division unit ('circular' or 'linear' or 'relu').",
                        type=str, default='linear')
    parser.add_argument("-i", "--input_image", help="image to transform", type=str, default=None)
    parser.add_argument("-e", "--epochs", help="Number of epochs between frames.", type=int, default=1)
    parser.add_argument("-x", "--mult", help="Output image dimension multiplier.", type=float, default=1.0)
    parser.add_argument("-m", "--model_file", help="model to load & continue.", type=str, default=None)
    parser.add_argument("-j", "--just_image", help="Just do an image, no training.", action='store_true', default=False)
    parser.add_argument("-b", "--border", help="Extrapolate outward from the original shape by this factor.",
                        type=float, default=0.0)
    parser.add_argument("-p", "--downsample", help="Downsample image by this factor, speeds up training at the cost of detail.", type=float, default=1.0)
    parser.add_argument('-l', "--learning_rate", help="Learning rate for the optimizer.", type=float, default=.01)
    parser.add_argument('-s', "--sharpness", help="Sharpness constant for activation function, e.g. f(x) = tanh(x*sharpness) for linear.", type=float, default=1000.0)
    parser.add_argument("--use_false_gradient", help="Use false gradient (tanh(cos_theta)) for LineLayer instead of sharp gradient.", action='store_true', default=False)
    parsed = parser.parse_args()

    kwargs = {'epochs_per_cycle': parsed.epochs, 'display_multiplier': parsed.mult,
              'border': parsed.border, 'sharpness': parsed.sharpness,
              'downsample': parsed.downsample, 'n_dividers': parsed.n_dividers,
              'just_image': parsed.just_image, 'n_hidden': parsed.n_hidden,
              'div_type': parsed.type, 'learning_rate': parsed.learning_rate}

    if parsed.input_image is not None:
        s = UIDisplay(image_file=parsed.input_image, **kwargs)
    else:
        if parsed.model_file is None:
            raise Exception("Need input image (-i) or model file (-m).")
        s = UIDisplay(state_file=parsed.model_file, **kwargs)

    s.run()
