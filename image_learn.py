import tensorflow as tf
import keras
import tempfile
import shutil
import cPickle as cp
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import numpy as np
import cv2
import re
import os
import logging
import argparse
import vispy.scene
import vispy.scene
import vispy.app
from vispy.scene.visuals import Image
from threading import Thread
from circular import CircleLayer
from linear import LineLayer


class ScaleInvariantImage(object):
    """
    Train a feed-forward neural network to approximate an image:
    learn f(x, y) = (r,g,b) using every pixel as a training example

    """
    def __init__(self, image=None, n_cores=0, state=None):

        if int(image is None) + int(state is None) != 1:
            raise Exception("Need input image, or state to restore.")

        self._image = image
        self._init_model(n_cores, state)
        self._make_train()
        self._n_cores = n_cores  # override

    def get_image(self):
        return self._image
        
    def pixel_to_coord(self, x, y):

        # scale to fit in [-1, 1] x [-1, 1]
        x = (x - self._image.shape[1]/2.0) * self._scale
        y = (y - self._image.shape[0]/2.0) * self._scale

        # add polar coords?
        #r = np.sqrt(x*x + y*y)
        #t = np.abs(np.arctan2(y, x))

        return x, y  # , r, t

    def _make_input(self, shape):
        x, y = np.meshgrid(np.arange(shape[1]),
                           np.arange(shape[0]))
        x = x * self._image.shape[1] / float(shape[1])
        y = y * self._image.shape[0] / float(shape[0])
        x, y = self.pixel_to_coord(x, y)
        logging.info( "Made inputs [%i, %i] spanning [%.3f, %.3f] and [%.3f, %.3f]."  % (x.shape[0], x.shape[1], np.min(x), np.max(x), np.min(y), np.max(y)))
        return np.hstack((x.reshape(-1,1), y.reshape(-1, 1)))#, r.reshape(-1,1)))

    def _calc_scale(self):
        big_dim = np.max(self._image.shape)
        self._scale = 2.0 / big_dim

    def _make_train(self):
        self._calc_scale()
        r, g, b = cv2.split(self._image)
        self._input = self._make_input(self._image.shape)
        self._output = np.hstack((r.reshape(-1,1), g.reshape(-1,1), b.reshape(-1,1)))/255.0
        logging.info( "Made input:  %s" % (self._input.shape,))
        logging.info( "Made output:  %s" % (self._output.shape,))
        logging.info( "Input spans:  [%f, %f]" % (np.min(self._input), np.max(self._input)))

    def get_session_and_graph(self):
        return self._tf_session, self._tf_graph

    def get_model(self):
        return self._model

    def _init_model(self, n_cores, state):
        if n_cores > 0:
            K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=n_cores,
                                                               inter_op_parallelism_threads=1)))

        self._tf_session = K.get_session()  # this creates a new session since one doesn't exist already.
        self._tf_graph = tf.get_default_graph()
        if state is not None:
            self._model = state['model']
            self._image = state['image']
            logging.info( "Model state restored")
            return

        """
        # Learn using ReLu inputs ...
        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self._n_hidden, activation=tf.nn.relu, use_bias=True,
                                  kernel_initializer='random_normal'),
            tf.keras.layers.Dense(3, activation=keras.layers.LeakyReLU(alpha=0.1), use_bias=True)])

        # ... or, learn using 2 layers of ReLu inputs
        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(2000, activation=tf.nn.relu, use_bias=True, kernel_initializer='random_normal'),
            tf.keras.layers.Dense(2000, activation=tf.nn.relu, use_bias=True, kernel_initializer='random_normal'),
            tf.keras.layers.Dense(3, use_bias=True)])


        self._model.compile(optimizer='adadelta',
                            loss='mean_squared_error',
                            metrics=['accuracy'])
        """

        # Learn using lines and or circles (functional api)
        input = Input((2,))
        # Use just a line layer, and some ReLu units to color the regions partitioned by the lines
        line_layer = LineLayer(500, input_shape=(2,))(input)
        #circle_layer = CircleLayer(100,input_shape=(2,))(input)
        middle = Dense(10, activation = tf.nn.relu, use_bias=True,
                       kernel_initializer='random_normal')(line_layer)
        #middle2 = Dense(20, activation = tf.nn.relu, use_bias=True,
        #               kernel_initializer='random_normal')(middle)
        """
        # Use both circles and lines in parallel
        circle_layer = CircleLayer(100,input_shape=(2,))(input)
        line_layer = LineLayer(100, input_shape=(2,))(input)
        middle = Dense(500, activation = tf.nn.relu, use_bias=True,
                      kernel_initializer='random_normal')(keras.layers.concatenate([circle_layer, line_layer]))
        """

        output = Dense(3, use_bias=True)(middle)
        self._model = Model(inputs=input, outputs=output)

        self._model.compile(loss='mean_squared_error',
                            optimizer='adadelta')

        logging.info( "INIT DONE")

    def train_epochs(self, n=1):
        self._model.fit(self._input, self._output, epochs=n, shuffle=True)

    def get_display_image(self, shape, margin=1.0):
        inputs = self._make_input(shape)  * margin
        logging.info( "Rescaled inputs to span [%.3f, %.3f] and [%.3f, %.3f]."  % (np.min(inputs[:,0]),
                                                                                     np.max(inputs[:,0]),
                                                                                     np.min(inputs[:,1]),
                                                                                     np.max(inputs[:,1])))
        rgb = self._model.predict(inputs)
        logging.info( "Display spans:  [%.3f, %.3f]" % (np.min(rgb), np.max(rgb)))
        img = cv2.merge((rgb[:,0].reshape(shape[:2]), rgb[:,1].reshape(shape[:2]), rgb[:,2].reshape(shape[:2])))
        n_clipped = np.sum(img<0) + np.sum(img>1)
        logging.info( "Display clipping:  %i (%.3f %%)" % (n_clipped,float(n_clipped)/img.size * 100.0))
        img[img<0.] = 0.
        img[img>1.] = 1.
        return img


def flip(img):
    return img[::-1,:,:]


def privatize(arg):
    return "_%s" % (arg, )


class VispyUI(object):
    _CUSTOM_LAYERS = {'CircleLayer': CircleLayer, 'LineLayer': LineLayer}

    # Variables possible for kwargs, use these defaults if missing from kwargs
    _DEFAULTS = {'n_cores': 0,
                 'epochs_per_cycle': 10,
                 'display_multiplier': 3,
                 'border': 1.0}

    def __init__(self, state_file=None, image=None, suffix=None, out_dir="output", **kwargs):
        # epochs_per_cycle=None, n_cores=None, display_multiplier=None, border=None):

        if int(image is None) + int(state_file is None) != 1:
            raise Exception("Need input image, or state to restore.")
        # state
        self._tempdir = tempfile.mkdtemp(prefix="imageStretch_")

        if state_file is not None:
            self._load_state(state_file)
        else:
            self._suffix = "" if suffix is None else "_%s" % (suffix,)
            self._out_dir = out_dir
            self._image = image
            n_cores = kwargs['n_cores'] if 'n_cores' in kwargs else self._DEFAULTS['n_cores']
            self._sim = ScaleInvariantImage(image, n_cores=n_cores, state=None)
            self._index = 0
            self._increment_run_number()

        # let these be overridden by command line if they're still none
        for arg in self._DEFAULTS:
            if arg in kwargs and kwargs[arg] is not None:
                setattr(self, privatize(arg), kwargs[arg])
            elif not hasattr(self, privatize(arg)):
                setattr(self, privatize(arg), self._DEFAULTS[arg])

        logging.info("Using args:")
        for arg in self._DEFAULTS:
            logging.info("\t%s: %s" % (arg, getattr(self, privatize(arg))))

        self._interactive = not kwargs['just_image']
        if kwargs['just_image']:
            self._write_image()
            return

        # set up VISPY display
        self._canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title="Route editor -- H for help")
        self._viewbox = self._canvas.central_widget.add_view()
        self._viewbox.camera = vispy.scene.cameras.PanZoomCamera(parent=self._viewbox.scene, aspect=1)
        self._add_graphics_elements()
        self.set_text_box_state(False)
        self._canvas.events.key_press.connect(self._on_key_press)
        self._canvas.events.close.connect(self._on_close)
        self._closing = False
        self._set_image(self._image / 255.0)
        self._worker = Thread(target=self._train_thread)

        self._worker.start()
        logging.info( "Started worker thread.")

    def is_interactive(self):
        return self._interactive

    def _on_close(self, event):
        self._closing = True
        logging.info ("Shutting down...")

    def _increment_run_number(self):
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)
        files = [f for f in os.listdir(self._out_dir) if f.startswith('output') and f.endswith('.png')]
        numbers = [int(re.search('^output_([0-9]+)_.+.png', f).groups()[0]) for f in files]
        self._run = np.max(numbers) + 1 if len(numbers) > 0 else 0

    def _save_state(self):
        logging.info("Saving state...")
        model_filename = "model_%i%s.tf" % (self._run, self._suffix)
        model_filepath = os.path.join(self._out_dir, model_filename)
        temp_filepath = os.path.join(self._tempdir, model_filename)
        self._sim.get_model().save(temp_filepath)
        with open(temp_filepath, 'r') as infile:
            model_bin = infile.read()
        model = {'model_bin': model_bin,
                 'epc': self._epochs_per_cycle,
                 'dm': self._display_multiplier,
                 'suffix': self._suffix,
                 'out_dir': self._out_dir,
                 'image': self._image,
                 'border': self._border,
                 'index': self._index,
                 'run': self._run}

        with open(model_filepath, 'w') as outfile:
            cp.dump(model, outfile)
        logging.info ("Wrote:  %s" % (model_filepath,))

    def _load_state(self, model_filepath):
        logging.info("loading state...")
        with open(model_filepath, 'r') as infile:
            state = cp.load(infile)
        self._epochs_per_cycle = state['epc']
        self._display_multiplier = state['dm']
        self._suffix = state['suffix']
        self._out_dir = state['out_dir']
        self._image = state['image']
        self._index = state['index'] + 1
        self._run = state['run']
        self._border = state['border']
        model_filename = "model_%i%s.tf" % (self._run, self._suffix)
        model_filepath = os.path.join(self._tempdir, model_filename)
        with open(model_filepath, 'w') as outfile:
            outfile.write(state['model_bin'])
        state['model'] = keras.models.load_model(model_filepath, custom_objects=self._CUSTOM_LAYERS)
        self._sim = ScaleInvariantImage(state=state)

    def _train_thread(self):
        tf_session, tf_graph = self._sim.get_session_and_graph()
        with tf_session.as_default():
            with tf_graph.as_default():

                while not self._closing:
                    for ep in range(self._epochs_per_cycle):
                        if self._closing:
                            break
                        self._sim.train_epochs(1)
                    self._save_state()
                    if self._closing:
                        break
                    img = self._write_image()
                    self._set_image(img)
                    self._index += 1
                    if self._closing:
                        break
        self._save_state()
        logging.info ("Close detected, shutting down worker thread.")
        logging.info ("Deleting work dir:  %s" % (self._tempdir,))
        shutil.rmtree(self._tempdir)

    def _write_image(self):

        img = self._sim.get_display_image(np.array(self._sim.get_image().shape) * self._display_multiplier,
                                          margin=self._border)

        out_filename = "output_%i%s_%.8i.png" % (self._run, self._suffix, self._index)
        out_path = os.path.join(self._out_dir, out_filename)
        while os.path.exists(out_path):
            self._index += 1
            out_filename = "output_%i%s_%.8i.png" % (self._run, self._suffix, self._index)
            out_path = os.path.join(self._out_dir, out_filename)

        cv2.imwrite(out_path, np.uint8(255 * img[:, :, ::-1]))
        logging.info ("Wrote:  %s" % (out_path,))
        return img

    def _add_graphics_elements(self):
        """
        Create the VISPY graphics objects
        """

        # Image
        self._image_object = Image(self._image, parent=self._viewbox.scene)
        self._image_object.set_gl_state('translucent', depth_test=False)
        self._image_object.order = 1
        self._image_object.visible = True

        self._text_box_width = 150
        self._text_box_height = 40
        self._text_box_offset = 10
        # Text background box in upper-left corner
        self._text_bkg_rect = vispy.scene.visuals.Rectangle([self._text_box_width / 2 + self._text_box_offset,
                                                             self._text_box_height / 2 + self._text_box_offset],
                                                            color=[0.1, 0.0, 0.0, .8],
                                                            border_color=[0.1, 0.0, 0.0],
                                                            border_width=2,
                                                            height=self._text_box_height,
                                                            width=self._text_box_width,
                                                            radius=10.0,
                                                            parent=self._canvas.scene)
        self._text_bkg_rect.set_gl_state('translucent', depth_test=False)
        self._text_bkg_rect.visible = True
        self._text_bkg_rect.order = 2

        # Text
        self._text = "?"
        self._text_pos = [self._text_box_offset + 10, self._text_box_offset + 10]
        self._text_obj = vispy.scene.visuals.Text(self._text,
                                                  parent=self._canvas.scene,
                                                  color=[0.9, 0.8, 0.8],
                                                  anchor_x='left',
                                                  anchor_y='top')
        self._text_obj.pos = self._text_pos
        self._text_obj.font_size = 18
        self._text_obj.visible = True
        self._text_obj.order = 3

    def set_text_box_state(self, state):
        self._text_bkg_rect.visible = state
        self._text_obj.visible = state

    def _change_text(self, new_text):
        if new_text is not None:
            self._text = new_text
        else:
            self._text = ""
        self._text_obj.text = self._text

    def _set_image(self, image=None):
        xmin, ymin = 0, 0
        xmax, ymax = image.shape[1], image.shape[0]
        self._viewbox.camera.set_range(x=(xmin, xmax), y=(ymin, ymax), z=None)
        self._image_object.set_data(flip(image))
        self._image_object.visible = True

    def _on_key_press(self, ev):
        if ev.key.name == 'Escape':
            self._canvas.close()
        else:
            self._change_text(ev.key.name)
            logging.info("Unknown keypress:  %s" % (ev.key.name,))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Learn an image.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_image", help="image to transform", type=str, default=None)
    parser.add_argument("-o", "--output_dir", help="Output frames to this folder.", type=str, default='output')
    parser.add_argument("-s", "--suffix", help="Suffix to put in filename.", type=str, default=None)
    parser.add_argument("-e", "--epochs", help="Number of epochs between frames.", type=int, default=None)
    parser.add_argument("-x", "--mult", help="Output image dimension multiplier.", type=int, default=None)
    parser.add_argument("-m", "--model_file", help="model to load & continue.", type=str, default='output')
    parser.add_argument("-c", "--cores", help="Use this many cores.", type=int, default=None)
    parser.add_argument("-j", "--just_image", help="Just do an image, no training.", action='store_true', default=False)
    parser.add_argument("-b", "--border", help="Extrapolate outward from the original shape by this factor.",
                        type=float, default=None)
    parsed = parser.parse_args()

    kwargs = {'epochs_per_cycle': parsed.epochs, 'display_multiplier': parsed.mult, 'suffix': parsed.suffix,
              'out_dir': parsed.output_dir, 'n_cores': parsed.cores, 'border': parsed.border,
              'just_image': parsed.just_image}

    if parsed.input_image is not None:
        img = cv2.imread(parsed.input_image)[:, :, ::-1]
        s = VispyUI(image=img, **kwargs)
    else:
        if parsed.model_file is None:
            raise Exception("Need input image (-i) or model file (-m).")
        s = VispyUI(state_file=parsed.model_file, **kwargs)
    if s.is_interactive():
        vispy.app.run()
