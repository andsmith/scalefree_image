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


class ScaleFreeImage(object):
    def __init__(self, training_data_file, layers=(100,100), batch_size=1024, monochrome=False):
        self.training_data_file = training_data_file
        self.layers = layers
        self.batch_size = batch_size
        self.monochrome = monochrome
        self._win_name = "Feed-forward network %s learning scale-free image." % (str(layers),)
        self._load_data()
        fact = 900 / max(self._img_shape[0], self._img_shape[1])
        x,y = make_input_grid(self._img_shape, resolution=fact)
        self._out_shape = x.shape
        self._disp_input = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))

        logging.info("Input shape: %s" % (self._disp_input.shape,))
        logging.info("Output shape: %s" % (self._out_shape,))
        logging.info("Image shape: %s" % (self._img_shape,))
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)

        self._init_model()
        self._frame = self._output_vec_to_image(self._output)
        self._refresh()

    def _load_data(self):
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



    def _output_vec_to_image(self, output_vec):
        if self._img_shape[2] == 1:
            img = output_vec.reshape((self._img_shape[0], self._img_shape[1]))
            img = np.clip(img*255.0, 0, 255).astype(np.uint8)
        else:
            img = output_vec.reshape((self._img_shape[0], self._img_shape[1], self._img_shape[2]))
            img = np.clip(img*255.0, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _init_model(self):
        inputs = Input(shape=(self._input.shape[1],))
        x = inputs
        for n in self.layers:
            x = Dense(n, activation='relu')(x)
        outputs = Dense(self._output.shape[1], activation='sigmoid')(x)
        self._model = Model(inputs=inputs, outputs=outputs)
        self._model.compile(optimizer='adam', loss='mse')
        logging.info("Initialized model with layers: %s" % (self.layers,))
        self._model.summary()

    def train_epochs(self, n=1, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        self._model.fit(self._input, self._output, epochs=n, batch_size=batch_size, verbose=True)
        output_pred = self._model.predict(self._input_orig )
        self._frame = self._output_vec_to_image(output_pred)
        self._refresh()

    def evaluate(self):
        loss = self._model.evaluate(self._input, self._output, verbose=0)
        logging.info("Evaluation loss: %f" % loss)
        return loss
    
    def _refresh(self):
        cv2.imshow(self._win_name, self._frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description='Train a feed-forward network to learn a scale-free image.')
    parser.add_argument('training_data_file', type=str, help='Path to the training data npz file.')
    parser.add_argument('-l', '--layers', type=int, nargs='+', default=[100, 100], help='List of hidden layer sizes.')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('-m', '--monochrome', action='store_true', help='Convert output to monochrome if true.')
    parser.add_argument('-e', '--epochs_per_update', type=int, default=100, help='Number of epochs to train per update.')

    args = parser.parse_args()

    sfi = ScaleFreeImage(args.training_data_file, layers=args.layers, batch_size=args.batch_size, monochrome=args.monochrome)
    while True:
        sfi.train_epochs(n=args.epochs_per_update)
        sfi.evaluate()
    logging.info("Training complete. Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()