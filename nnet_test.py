"""
Load the training set from a npz file and train a standard feed-forward network to create a scale-free iamge.
This is a test script to see how well a standard network can learn a scale-free image.


"""
import numpy as np
import cv2
import logging
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from util import make_input_grid
from tensorflow.keras.models import Model

import os

class ScaleFreeImage(object):
    def __init__(self, training_data_file, layers=(100,100), batch_size=1024, monochrome=False,
                    display_multiplier=1.0, learning_rate=0.001, frame_dir=None, weights_file=None):
        self._frame_dir = frame_dir
        self.learning_rate = learning_rate
        self._shutdown = False
        self._frame_num = 0
        self._disp_mul = display_multiplier
        self.training_data_file = training_data_file
        self.layers = layers
        self.batch_size = batch_size
        self.monochrome = monochrome
        
        self._win_name = "Feed-forward network %s learning scale-free image." % (str(layers),)
        self._load_data(display_multiplier)
        fact = 900 / max(self._img_shape[0], self._img_shape[1])
        x,y = make_input_grid(self._img_shape, resolution=fact)
        self._out_shape = x.shape
        self._disp_input = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
        self._train_path = os.path.dirname(self.training_data_file)

        logging.info("Input shape: %s" % (self._disp_input.shape,))
        logging.info("Output shape: %s" % (self._out_shape,))
        logging.info("Image shape: %s" % (self._img_shape,))
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)

        self._init_model(weights_file)

        self._refresh()

    def _get_filename(self, which='weights'):
        """
        weights: <path_to_training_data>/<training_data_file>_<layers>.npz
        output frame: <frame_dir>/<training_data_file>_<layers>_frame_0000001.png
        """
        training_data_file = os.path.basename(self.training_data_file).split('.')[0]
        layer_str = "-".join(map(str, self.layers))
        file_base = "%s_%s" % (training_data_file, layer_str)
        if which == 'weights':
            return os.path.join(self._train_path, "%s.weights.npz" % (file_base,))
        elif which == 'frame':
            return os.path.join(self._frame_dir, "%s_frame_%05d.png" % (file_base, self._frame_num))
        else:
            raise ValueError("Unknown filename type: %s" % which)

    def _load_data(self, display_multiplier):
        data = np.load(self.training_data_file)
        self._input = data['input']
        self._output = data['output']
        if self.monochrome and self._output.shape[1] == 3:
            self._output = np.mean(self._output, axis=1, keepdims=True)
        self._img_shape = data['img_shape']
        logging.info("Loaded training data from %s" % self.training_data_file)
        logging.info("Input shape: %s" % (self._input.shape,))
        logging.info("Output shape: %s" % (self._output.shape,))
        self._input_orig = self._input.copy()
        self._output_orig = self._output.copy()
        #self._inputs_test = make_input_grid(self._img_shape, resolution=1.0)

        x, y = make_input_grid(self._img_shape, resolution=display_multiplier)
        self._display_shape = x.shape[0], x.shape[1], 3
        self._display_input = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        # self._inputs_test = make_input_grid(self._img_shape, resolution=1.0)


    def _output_vec_to_image(self, output_vec, img_shape):
        if img_shape[2] == 1:
            img = output_vec.reshape((img_shape[0], img_shape[1]))
            img = np.clip(img*255.0, 0, 255).astype(np.uint8)
        else:
            img = output_vec.reshape((img_shape[0], img_shape[1], img_shape[2]))
            img = np.clip(img*255.0, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _init_model(self, weights_file=None):
        inputs = Input(shape=(self._input.shape[1],))
        x = inputs
        for n in self.layers:
            x = Dense(n, activation='relu')(x)
        outputs = Dense(self._output.shape[1], activation='sigmoid')(x)
        self._model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if weights_file is not None:
            weights = np.load(weights_file)
            wts = [weights[f'arr_{i}'] for i in range(len(weights.files))]
            self._model.set_weights(wts)
            logging.info(f"Loaded model weights from {weights_file}")
        self._model.compile(optimizer=optimizer, loss='mse')
        logging.info("Initialized model with layers: %s" % (self.layers,))
        self._model.summary()

    def train_epochs(self, n=1, batch_size=None):
        self._frame_num += 1
        if batch_size is None:
            batch_size = self.batch_size
        self._model.fit(self._input, self._output, epochs=n, batch_size=batch_size, verbose=True)
        return self._refresh()

    def evaluate(self):
        loss = self._model.evaluate(self._input, self._output, verbose=0)
        logging.info("Evaluation loss: %f" % loss)
        return loss
    
    def _refresh(self):
        output_pred = self._model.predict(self._display_input)
        self._frame = self._output_vec_to_image(output_pred, self._display_shape)
        
        if self._frame.shape[:2] != self._out_shape:
            # resize window to match image:
            cv2.resizeWindow(self._win_name, self._frame.shape[1], self._frame.shape[0])
            self._out_shape = self._frame.shape[:2]


        cv2.imshow(self._win_name, self._frame)
        k = cv2.waitKey(1) & 0xFF
    
        if self._frame_dir is not None:
            frame_name = self._get_filename('frame')

            if not os.path.exists(self._frame_dir):
                os.makedirs(self._frame_dir)
            cv2.imwrite(frame_name, self._frame)
            logging.info(f"Saved frame to {frame_name}")

        if k == 27 or k == ord('q'):
            return True
        return False

    def save(self):
        weights_filename = self._get_filename('weights')
        weights = self._model.get_weights()
        np.savez(weights_filename, *weights)
        logging.info(f"Saved model weights to {weights_filename}")

def get_args():
    
    import argparse

    parser = argparse.ArgumentParser(description='Train a feed-forward network to learn a scale-free image.')
    parser.add_argument('training_data_file', type=str, help='Path to the training data npz file.')
    parser.add_argument('-l', '--layers', type=int, nargs='+', default=[100, 100], help='List of hidden layer sizes.')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('-m', '--monochrome', action='store_true', help='Convert output to monochrome if true.')
    parser.add_argument('-e', '--epochs_per_update', type=int, default=100, help='Number of epochs to train per update.')
    parser.add_argument('-x', '--display_multiplier', type=float, default=1.0, help='Display scaling multiplier.')
    parser.add_argument('-r', '--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('-f', '--save_frames', type=str)
    parser.add_argument('-w', '--weights_file', type=str, default=None, help='Path to load model weights from.')


    args = parser.parse_args()
    return args
def app(args):
    
    sfi = ScaleFreeImage(args.training_data_file, layers=args.layers, batch_size=args.batch_size, monochrome=args.monochrome,
                            display_multiplier=args.display_multiplier, learning_rate=args.learning_rate, frame_dir=args.save_frames,
                            weights_file=args.weights_file)
    done = False
    while not done:
        if sfi.train_epochs(n=args.epochs_per_update):
            done=True
        sfi.save()
    logging.info("Training complete. Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    app(args)