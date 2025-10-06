"""
Load the training set from a npz file and train a standard feed-forward network to create a scale-free iamge.
This is a test script to see how well a standard network can learn a scale-free image.

For training, interpolate the image range 

"""
import numpy as np
import cv2
import logging
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from util import make_input_grid
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import time
from threading import Thread, Lock, Event
import json
import matplotlib.pyplot as plt
from scipy import ndimage
from copy import deepcopy
import os

class ScaleFreeImage(object):
    def __init__(self, image_filename, out_dir, layers=(100,100), batch_size=1024, n_train_sample=16384,
                    display_multiplier=1.0, learning_rate=0.001, weights_file=None, epochs_per_cycle=100):
        self.learning_rate = learning_rate
        self._shutdown = False
        self._disp_mul = display_multiplier
        self.n_train = n_train_sample
        self.layers = layers
        self.batch_size = batch_size
        self.out_dir= out_dir
        self._win_name = "Feed-forward network %s learning scale-free image." % (str(layers),)
        self._filename_base =self._get_base_filename(image_filename)
        self._epoch_over = Event()
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        
        self._load_data(image_filename)
        
        self.aspect = self.train_image.shape[1] / self.train_image.shape[0]
        if self.aspect > 1.0:
            self.xlim = -1.0, 1.0
            self.ylim = -1.0 / self.aspect, 1.0 / self.aspect
        else:
            self.ylim = -1.0, 1.0
            self.xlim = -1.0 * self.aspect, 1.0 * self.aspect
        
        
        self.cycle_ind = 0
        self.epoch_ind=0
        self.epochs_per_cycle = epochs_per_cycle
        self._input, self._output = None, None  # set before each epoch
        self._update_lock = Lock()
        
        self._show_train_img = False  # in GUI show this instead of the latest output image  (Toggle with 'T' key)
        self._show_train_sample_xy = False  # in GUI show the sampled training positions in the image (toggle with 'I' key)

        logging.info("Input shape: %s" % (self.train_image.shape,))
        logging.info("Output shape: %s" % (self.output_image.shape,))
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)

        self._history = [] # one entry per update cycle, keys "learning_rate" (float), "losses" (1 list per epoch of minibatch losses (floats))

        if weights_file is None:
            weights_file = self._get_filename('weights')
            if not os.path.exists(weights_file):
                weights_file = None
                logging.info("No existing weights file found at %s. Starting new model." % (weights_file,))
            else:
                logging.info("Resuming training from existing weights file: %s" % (weights_file,))

        self._init_model(weights_file)
        
    def _get_base_filename(self, image_filename):
        """
        base is the image name without path or extension.
        also if it has _train_image.[ext] at the end, remove that too
        """
        base = os.path.basename(image_filename)
        name, _ = os.path.splitext(base)
        if name.endswith("_train_image"):
            name = name[:-len("_train_image")]
        return name
    
    
        return os.path.join(self.out_dir, name)
    def _write_metadata(self):
        meta_file = self._get_filename('metadata')
        with open(meta_file, 'w') as f:
            json.dump(self._history, f)
        logging.info(f"Saved training metadata to {meta_file}")
        
    def _read_metadata(self):
        
        meta_file = self._get_filename('metadata')
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                self._history = json.load(f)
            logging.info(f"Loaded training metadata from {meta_file}, resuming from cycle {len(self._history)}")
            self.cycle_ind = len(self._history)
        else:
            logging.info(f"No metadata file found at {meta_file}")

    def _get_filename(self, which='weights'):
        """
        weights: <path_to_training_data>/<training_data_file>_<layers>.npz
        output frame: <frame_dir>/<training_data_file>_<layers>_frame_0000001.png
        """
        layer_str = "-".join(map(str, self.layers))
        file_base = "%s_%s" % (self._filename_base, layer_str)
        if which == 'weights':
            fname="%s.model_weights.npz" % (file_base,)
        elif which == 'frame':
            fname="%s_frame_%08d.png" % (file_base, self.cycle_ind)
        elif which == 'metadata':
            fname="%s.metadata.json" % (file_base,)
        elif which == 'train_image':
            fname="%s_training_image.png" % (file_base,)
        else:
            raise ValueError("Unknown filename type: %s" % which)
        return os.path.join(self.out_dir, fname)
    
    
    def _load_data(self, image_filename):

        self.train_image = cv2.imread(image_filename).astype(np.float32) / 255.0
        output_shape_hw = (np.array((self.train_image.shape[0]* self._disp_mul, self.train_image.shape[1]* self._disp_mul,3)) ).astype(int)
        
        self._train_color_interps = {}
        
        
        if self.train_image.shape[2] !=3:
            raise ValueError("Input image must have 3 channels.")
        self.output_shape = output_shape_hw[0], output_shape_hw[1], 3
            
        self.output_image = np.zeros(self.output_shape, dtype=np.float32)
        

        logging.info("Loaded training data from %s"% image_filename)
        logging.info("Input shape: %s" % (self.train_image.shape,))
        logging.info("Output shape: %s" % (self.output_image.shape,))
        
        
        
        
        
        train_filename = self._get_filename('train_image')
        cv2.imwrite(train_filename, (self.train_image*255).astype(np.uint8))
        logging.info(f"Saved training image to {train_filename}")

        # For generating the output image:
        x, y = make_input_grid(output_shape_hw, resolution=1.0) # already magnifide
        self._output_xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        
        # For defining color positions for training input sampling (interpolated xy positions):
        x, y = make_input_grid(self.train_image.shape[:2], resolution=1.0)
        self._train_xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        


    def _output_vec_to_image(self, output_vec, img_shape):
        if img_shape[2] == 1:
            img = output_vec.reshape((img_shape[0], img_shape[1]))
            img = np.clip(img*255.0, 0, 255).astype(np.uint8)
        else:
            img = output_vec.reshape((img_shape[0], img_shape[1], img_shape[2]))
            img = np.clip(img*255.0, 0, 255).astype(np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _init_model(self, weights_file=None):
        n_input,n_output = 2,3
        inputs = Input(shape=(n_input,)) # x,y input
        x = inputs
        for n in self.layers:
            activation = 'relu' # if n > 0 else 'sigmoid'
            x = Dense(n, activation=activation, use_bias=True)(x)
        outputs = Dense(n_output, activation='sigmoid')(x)
        self._model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        if weights_file is not None:
            weights = np.load(weights_file)
            wts = [weights[f'arr_{i}'] for i in range(len(weights.files))]
            self._model.set_weights(wts)
            logging.info(f"Loaded model weights from {weights_file}")
            
            self._read_metadata()
            
        self._model.compile(optimizer=optimizer, loss='mse')
        
        logging.info("Initialized model with layers: %s" % (self.layers,))
        self._model.summary()


    def save(self):
        weights_filename = self._get_filename('weights')
        weights = self._model.get_weights()
        np.savez(weights_filename, *weights)
        logging.info(f"Saved model weights to {weights_filename}")

        self._write_metadata()
        
        
    def _sample_training_input(self):
        """ 
        """
        if self.n_train == 0:
            self._input = self._train_xy
            r = self.train_image[:,:,0].flatten()
            g = self.train_image[:,:,1].flatten()
            b = self.train_image[:,:,2].flatten()
            self._output = np.vstack((r,g,b)).T
            order = np.random.permutation(self._input.shape[0])
            self._input = self._input[order]
            self._output = self._output[order]
            
        else:
            x = np.random.uniform(self.xlim[0], self.xlim[1], size=(self.n_train,1))
            y = np.random.uniform(self.ylim[0], self.ylim[1], size=(self.n_train,1))
            x_pixel = ((x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.train_image.shape[1]-1))
            y_pixel = ((y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.train_image.shape[0]-1))
            
            self._input = np.hstack((x,y))
            
            
            r = ndimage.map_coordinates(self.train_image[:,:,0], [y_pixel.flatten(), x_pixel.flatten()], order=1)
            g = ndimage.map_coordinates(self.train_image[:,:,1], [y_pixel.flatten(), x_pixel.flatten()], order=1)
            b = ndimage.map_coordinates(self.train_image[:,:,2], [y_pixel.flatten(), x_pixel.flatten()], order=1)
            self._output = np.vstack((r,g,b)).T

    def _train_loop(self,verbose=False,n_cycles=-1):
        n_new_cycles = 0
        while not self._shutdown:  # Loop over update cycles:
            logging.info("Starting training cycle %d (%i this run), training on %i samples for %i epochs with batch size %i at learning rate %.6f"%(
                self.cycle_ind, n_new_cycles, self.n_train, self.epochs_per_cycle, self.batch_size, self.learning_rate))
            if n_cycles > -1 and n_new_cycles >= n_cycles:
                break
            
            # TRAIN ONE CYCLE:
            start_time = time.time()
            loss_tracker = BatchLossCallback()
            for epoch in range(self.epochs_per_cycle):
                if self._shutdown:
                    break
                self.epoch_ind = epoch
                self._sample_training_input()
                loss =self._model.fit(self._input, 
                                self._output, 
                                epochs=1,
                                batch_size=self.batch_size,
                                verbose=verbose, 
                                callbacks=[loss_tracker])
                print("\tEpoch %d loss: %.6f"%(epoch, loss.history['loss'][0]))
            cycle_info= {'losses': loss_tracker.losses,'learning_rate': self.learning_rate}    
            
            # RECOMPUTE OUTPUT IMAGE:

            output_pred = self._model.predict(self._output_xy, verbose=verbose, batch_size=32768)
            output_image = self._output_vec_to_image(output_pred, self.output_shape)
            # SAVE STATE:
            end_time = time.time()
            cycle_time = end_time - start_time
            #logging.info(f"Completed training cycle {self.cycle_ind} in {cycle_time:.2f} seconds.")
            
            frame_name = self._get_filename('frame')
            cv2.imwrite(frame_name, output_image)
            #logging.info(f"Saved output image to {frame_name}")
            cycle_info['cycle_time'] = cycle_time
            cycle_info['frame_file'] = frame_name
            
            with self._update_lock:
                self._history.append(cycle_info)
                self.output_image = output_image
                
            self.save()
            self.cycle_ind += 1
            n_new_cycles += 1
            
        logging.info("Training thread exiting.")
        
        
    def run(self, debug_mode=False):
        """"""
        if debug_mode:
            self._train_loop(n_cycles=1)  # Debug, no thread
            # Save training image
        else:
            train_thread = Thread(target=self._train_loop)
            train_thread.start()    
        
        # GUI loop
        
        
        ""
        
        artists = {'image': None,
                   'train_xy': None,
                   'learning_rate' : None, 
                   'minibatch_loss': None,
                   'epoch_loss': None,'cycle_loss': None}
        while True:
            with self._update_lock:
                frame = self.output_image.copy() if self.output_image is not None else np.zeros(self.output_shape, dtype=np.uint8)
                losses = deepcopy(self._history)
                
            cv2.imshow(self._win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._shutdown = True
                break


class BatchLossCallback(Callback):
    """
    Get loss vector (for all minibatches in an epoch) for every epoch
    """

    def __init__(self):
        """
        Record the losses from each minibatch & epoch for this cycle. (list of lists)
        
        If dry run, make up fake data, appropriate for the number of minibatches/epoch, and number of epochs
        """
        super().__init__()
        
        self.losses = []  # list for each epoch of all minibatch losses
            

    def on_epoch_begin(self, epoch, logs=None):
        self.losses.append([])  # new epoch    

    def on_train_batch_end(self, batch, logs=None):
        self.losses[-1].append(logs['loss'])
        

def get_args():
    
    import argparse

    parser = argparse.ArgumentParser(description='Train a feed-forward network to learn a scale-free image.')
    parser.add_argument('project_path', type=str, help='Path to the project directory (all output files).')
    parser.add_argument('-i', '--image_file', type=str, help='Path to the jpg/png image file to learn (required if project path contains no model to resume).'+
                        "WARNING:  if resuming training, this replaces project_path/training_image.png")
    parser.add_argument('-l', '--layers', type=int, nargs='+', default=[100, 100], help='List of hidden layer sizes.')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('-t', '--num_train', type=int, default=16384, help='Number of x,y positions to sample for each epoch (or 0 to just use pixel positions).')
    
    parser.add_argument('-e', '--epochs_per_update', type=int, default=100, help='Number of epochs to train per update cycle.')
    parser.add_argument('-x', '--display_multiplier', type=float, default=1.0, help='The output frame will be this X the size of the training image.')
    parser.add_argument('-r', '--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('-w', '--weights_file', type=str, default=None, help='Path to load model weights from.')
    

    args = parser.parse_args()
    
    return args

def app():

    args = get_args()
    sfi = ScaleFreeImage(image_filename=args.image_file,
                         out_dir = args.project_path,
                         layers=args.layers, 
                         batch_size=args.batch_size, 
                         epochs_per_cycle=args.epochs_per_update,
                         display_multiplier=args.display_multiplier,
                         learning_rate=args.learning_rate,
                         weights_file=args.weights_file,
                         n_train_sample=args.num_train)
    sfi.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()