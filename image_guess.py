"""
Like image_learn.py, but focused on testing lots of initializations quickly before investing in backpropagation.

General outline:

    A) These two run in separate threads, with their own pools of multiprocessing workers, etc.
    
        - CANDIDATE GENERATION:   Continually generate new random initializations ("candidates"), train for a small amount of epochs (the
          "audition"), and evaluate their loss.  This is a proxy to guess if they'll end up especially successful (TODO: validate this assumption). 
          Maintain a set of the top K candidates, the "talent pool" the trainees are recruited from. (see part B)
          
        - TRAINEE IMPROVEMENT:  Continually train a small number of the most promising candidates to further reduce the
          loss. 
          
        (producer-consumer pattern)

    B) Every D cycles of training, do these steps for a "draft" (updates to the processes in part A):
         * if any candidate in the talent pool has a loss that is less than any trainee's current loss, give the candidate
           the trainee's spot and discard the trainee.
         * With small probability, move the top candidate from the last cycle to the worst trainee's spot regardless of loss.
        

    C) There will be two sections for plots, one for candidates and one for trainees.
    
        
        The CANDIDATE plot will show:
        
            * the best candidate's output image after the audition training (best over all time) and its loss.
            * Histogram showing the distribution of all candidates' lossses as they accumulate.
            * Vertical red lines (?) on the histogram for losses of candidates in the talent pool.        
        
        The TRAINEE plot will show:
        
            * the best trainee's current output image, with moving divider units overlaid (like in image_learn.py)
            * all trainees' loss histories (as in image_learn.py)
            * trainee statistics as they accumulate:
                - How many draft cycles they survive.
                - Trainee loss histogram

    D) Specify all parameters as with image_learn.py, with the addition of:
    
         -D --draft_cycles [int]      # of cycles between drafts (default 5)
         -N --n_workers [int] [int]   # [number of trainee processes] [number of candidate processes]
         -L --candidate_luck [float]  # probability of randomly promoting the top candidate to a trainee spot (default 0.05)
         -U --audition_epochs [int]   # number of epochs to train each candidate during audition (default 50)
    
         Talent pool is set to 10. (make param?)
"""

import numpy as np
import cv2
import os
from image_learn import UIDisplay, ScaleInvariantImage, get_args
import logging
import threading
import matplotlib.pyplot as plt
import argparse

class GuesserUIDisplay(object):
    def __init__(self, save_path, image_file, cand_json, trainee_json, draft_cycles=5, candidate_luck=0.05, 
                 audition_epochs=50, n_candidates=50, n_trainees=5):
        self.save_path = save_path
        self.image_file = image_file
        self.cand_json = cand_json
        self.trainee_json = trainee_json
        self.draft_cycles = draft_cycles
        self.candidate_luck = candidate_luck
        self.audition_epochs = audition_epochs
        self.n_candidates = n_candidates
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            logging.info(f"Created save directory: {self.save_path}")
        
        self._shutdown = False
        self._draft = 0
        self._load_params()
        self._cand_thread = threading.Thread(target=self._candidate_thread_p)
        self._trainee_thread = threading.Thread(target=self._trainee_thread_p)
        
    def _load_params(self):
        import json
        with open(self.cand_json, 'r') as f:
            self.cand_params = json.load(f)
        with open(self.trainee_json, 'r') as f:
            self.trainee_params = json.load(f)
        self.trainee_params['cycles']=1
        self.cand_params['nogui']=True
        self.trainee_params['nogui']=True
        logging.info(f"Loaded candidate params from {self.cand_json}:")
        logging.info(f"Loaded trainee params from {self.trainee_json}:")
        
    def _trainee_thread_p(self):
        pass
    def _candidate_thread_p(self):
        pass   
    
    def _plot_key_callback(self, event):
        if event.key == 'q' or event.key == 'escape':
            logging.info("Quit key pressed, shutting down...")
            self._shutdown = True 
    
    def run(self):
        logging.info("Starting candidate and trainee threads...")
        self._cand_thread.start()
        self._trainee_thread.start()
        logging.info("Starting UI loop...")
        fig, axes = plt.subplots(2,2) # temp, so we can exit with keypress
        plt.connect('key_press_event', self._plot_key_callback)    
        
        plot_cycles=0
        
        while not self._shutdown:
            # update plots
            plt.draw()
            plt.pause(0.1)
            plot_cycles += 1
            
                
        logging.info("UI loop exiting, waiting for threads to finish...")        
        self._cand_thread.join()
        self._trainee_thread.join()
        logging.info("All threads and processes finished.")
        

def get_args():
    parser = argparse.ArgumentParser(description='Guess a large number of network weights, only train the best.')
    parser.add_argument('save_path', type=str, nargs='?', default=None, help='Path to the save directory.')
    parser.add_argument('input_image', type=str, nargs='?', default=None, help='Path to the input image file.')
    parser.add_argument('candidate_params_file', type=str, nargs='?', default=None, help='Path to the candidate parameter json file.')
    parser.add_argument('training_params_file', type=str, nargs='?', default=None, help='Path to the training parameter json file.')
    parser.add_argument('-d', '--draft_cycles', type=int, default=5, help="Number of cycles between drafts.")
    parser.add_argument('-l', '--candidate_luck', type=float, default=0.05, help="Probability of randomly promoting the top candidate to a trainee spot.")
    parser.add_argument('-u', '--audition_epochs', type=int, default=50, help="Number of epochs to train each candidate during audition.")
    parser.add_argument('-c', '--n_candidates', type=int, default=50, help="Number of candidates to maintain in the talent pool (1 per candidate process).")
    parser.add_argument('-t', '--n_trainees', type=int, default=5, help="Number of trainee worker processes to run.")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parsed = get_args()

    if parsed.input_image is None and parsed.model_file is None and parsed.test_image is None:
        raise Exception("Need input image (-i) to start training, or model file (-m) to continue.")
    
    s = GuesserUIDisplay(save_path=parsed.save_path, image_file=parsed.input_image,
                         cand_json=parsed.candidate_params_file, trainee_json=parsed.training_params_file,
                         draft_cycles=parsed.draft_cycles, candidate_luck=parsed.candidate_luck,
                         audition_epochs=parsed.audition_epochs, n_candidates=parsed.n_candidates,
                         n_trainees=parsed.n_trainees)
    s.run()
