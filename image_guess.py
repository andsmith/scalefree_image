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
from tempfile import mkdtemp
from multiprocessing import Process, Queue, Value
import time
import random
import shutil
import uuid
import pickle
from threading import Thread

CAND_DIR = "candidates"
TALENT_DIR = "talent"


def _make_candidates(cand_params, image_file, cand_dir, cand_queue, shutdown_flag, num_to_make = -1):
    """
    Process to continually generate candidates.
    :param cand_params:  parameters for candidate UIDisplay
    :param image_file:  path to the image file
    :param cand_dir:  directory to save candidate models
    :param cand_queue:  Queue to put candidates in when ready for audition
    :param shutdown_flag:  threading.Event to signal shutdown
    :param num_to_make:  number of candidates to generate, -1 to run indefinitely, 0 exit immediately (for debugging)
    :returns: number of candidates generated
    """
    print("Starting candidate generation process...77777777777777777777777777777777777777777")
    n_cand = 0
    while not shutdown_flag.value and (num_to_make < 0 or n_cand < num_to_make):
        prefix = str(uuid.uuid4())[:8]
        model_save_info = {'path': cand_dir, 'prefix': prefix}
        cand_params['frame_dir'] = None # don't save frames for candidates
        cand_params['image_file'] = image_file
        cand = UIDisplay(model_save_info=model_save_info, **cand_params)
        loss, model_file_path, meta_file_path = cand._train_thread(max_iter = cand_params['epochs_per_cycle'],fast=True)
        out_img = cand._output_image
        model_file_rel = os.path.basename(model_file_path)
        meta_file_rel = os.path.basename(meta_file_path)
        candidate_file = "%s_candidate.pkl" % (prefix,)
        candidate = {
            'candidate_file': candidate_file,
            'params': cand_params,
            'model_file': model_file_rel,
            'meta_file': meta_file_rel,
            'dir': cand_dir,
            'loss': loss,
            'output_image': out_img
        }
        candidate_file = os.path.join(cand_dir, candidate_file)
        with open(candidate_file, 'wb') as f:
            pickle.dump(candidate, f)
        logging.info(f"Generated candidate {candidate_file} with loss {loss:.4f}")
        
        # Put the candidate in the queue
        cand_queue.put(candidate)
        n_cand += 1
        
        
    return n_cand


class GuesserUIDisplay(object):
    def __init__(self, save_path, image_file, cand_json, trainee_json, draft_cycles=5, candidate_luck=0.05, 
                 audition_epochs=50, n_cand_procs=5, n_trainee_procs=5, max_candidates=-1):
        self.save_path = save_path
        self.image_file = image_file
        self.cand_json = cand_json
        self.trainee_json = trainee_json
        self.draft_cycles = draft_cycles
        self._max_candidates = max_candidates
        self.candidate_luck = candidate_luck
        self.audition_epochs = audition_epochs
        self.n_cand_procs = n_cand_procs
        self._history = {'losses': []}  # history of all candidate losses
        self.n_trainee_procs = n_trainee_procs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            logging.info(f"Created save directory: {self.save_path}")
        
        self._shutdown = Value('b', False)  # signal shutdown to threads and processes
        self._draft_num = 0  # number of drafts completed
        
        # Keep the N best candidates of all time (not used in a draft yet)
        self.talent_pool_size = 100  # number of candidates to keep in the talent pool
        self._cand_queue = Queue()  # Puts candidates ready for audition here (reference to model file in temp dir, with loss/output image).
        # candidate thread produces for this queue, trainee thread consumes from it.
        

        self._talent_pool = []  # list of dicts (candidate info) for current talent pool, in increasing loss order.
        
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
        
    def _load_talent_pool(self):
        """
        Scan candidates folder for all *_candidate.pkl files, load them, and keep the best K in the talent pool.
        """
        talent_dir = os.path.join(self.save_path, 'talent')
        if not os.path.exists(talent_dir):
            os.makedirs(talent_dir)
            logging.info(f"Created talent directory: {talent_dir}")
            self._talent_pool = []
        else:
            for filename in os.listdir(talent_dir):
                if filename.endswith('_candidate.pkl'):
                    with open(os.path.join(talent_dir, filename), 'rb') as f:
                        talent = pickle.load(f)
                        self._talent_pool.append(talent)
            self._talent_pool.sort(key=lambda x: x['loss'])
            self._talent_pool = self._talent_pool[:self.talent_pool_size]
            logging.info(f"Loaded talent pool with {len(self._talent_pool)} candidates.")

    def _candidate_thread_p(self, exit_after = -1):
        logging.info("Starting candidate generation thread.")
        # Start candidate generation process(es):
        candidate_dir= os.path.join(self.save_path, 'candidates')
        talent_dir = os.path.join(self.save_path, 'talent')

        if not os.path.exists(candidate_dir):
            os.makedirs(candidate_dir)
            logging.info(f"Created candidate directory: {candidate_dir}")
            
        if not os.path.exists(talent_dir):
            os.makedirs(talent_dir)
            logging.info(f"Created talent directory: {talent_dir}")
            
        self._load_talent_pool()
        
        logging.info(f"Candidate temporary directory: {candidate_dir}")
    
        # start candidate processes
        if self.n_cand_procs > 1:
            logging.info(f"Starting {self.n_cand_procs} candidate generation processes.")
            producer_procs = [Process(target=_make_candidates, args=(self.cand_params, self.image_file, candidate_dir, self._cand_queue, self._shutdown, 0))
                            for _ in range(self.n_cand_procs)]
            for p in producer_procs:
                p.start()
            logging.info(f"Started {self.n_cand_procs} candidate generation processes.")
        else:
            logging.info("Starting single candidate generation thread.")
            self._cand_thread = Thread(target=_make_candidates, args=(self.cand_params, self.image_file, candidate_dir,
                                                                        self._cand_queue, self._shutdown, exit_after))
            self._cand_thread.start()
            
            logging.info("Single candidate generation process finished.")
    
        def _remove_candidate_files(candidate):
            
            os.remove(os.path.join(candidate_dir, candidate['model_file']))
            os.remove(os.path.join(candidate_dir, candidate['meta_file']))
            os.remove(os.path.join(candidate_dir, candidate['candidate_file']))
            logging.info(f"Removed candidate files for candidate with loss {candidate['loss']:.4f}.")

        def _evict_from_pool(talent):
            os.remove(os.path.join(talent_dir, talent['model_file']))
            os.remove(os.path.join(talent_dir, talent['meta_file']))
            os.remove(os.path.join(talent_dir, talent['candidate_file']))
            logging.info(f"Removed talent with loss {talent['loss']:.4f} from talent pool.")
            

        n_proc=0
        n_added=[0]
        
        def _move_to_talent_pool(candidate):
            """
            Move files over, change filesnames in dict, return updated dict.
            """
            # Move model and meta files:
            new_model_file = os.path.join(talent_dir, os.path.basename(candidate['model_file']))
            new_meta_file = os.path.join(talent_dir, os.path.basename(candidate['meta_file']))
            new_candidate_file = os.path.join(talent_dir, os.path.basename(candidate['candidate_file']))
            
            shutil.move(os.path.join(candidate['dir'], candidate['model_file']), new_model_file)
            shutil.move(os.path.join(candidate['dir'], candidate['meta_file']), new_meta_file)
            shutil.move(os.path.join(candidate['dir'], candidate['candidate_file']), new_candidate_file)
            talent = candidate.copy()
            talent['n_added'] = n_added[0]
            talent['model_file'] = new_model_file
            talent['meta_file'] = new_meta_file
            talent['candidate_file'] = new_candidate_file
            
            # re-write the candidate file so it has the new value for n_proc:
            with open(new_candidate_file, 'wb') as f:
                pickle.dump(talent, f)
            logging.info(f"Moved candidate with loss {talent['loss']:.4f} to talent pool, 3 files moved.")
            n_added[0] += 1
            return talent
        
        
        history_file = os.path.join(self.save_path, 'candidate_history.pkl')
        if os.path.exists(history_file):
            with open(history_file, 'rb') as f:
                hist = pickle.load(f)
            logging.info(f"Loaded candidate history with {len(hist['losses'])} losses from {history_file}")
            self._history = hist
        else:
            self._history = {'losses': []}
            if len(self._talent_pool) > 0:
                logging.warning("Talent pool not empty but no history file found, starting new history, keeping existing talent pool.")
            
            
        def _update_history(candidate):
            candidate['n_proc'] = n_proc
            self._history['losses'].append(candidate['loss'])
            self._history['n_added'] = n_added[0]
            self._history['n_proc'] = n_proc
            history_file = os.path.join(self.save_path, 'candidate_history.pkl')
            with open(history_file, 'wb') as f:
                pickle.dump(self._history, f)
            logging.info(f"Saved candidate history with {len(self._history['losses'])} losses to {history_file}")
            return
        

        # Monitor the queue, when something comes in, see if it should be added to the talent pool:
        while not self._shutdown.value and (exit_after < 0 or n_proc < exit_after):
            candidate = self._cand_queue.get()  # wait for a candidate
            candidate['n_proc'] = n_proc  # order received in queue
            
            _update_history(candidate)
            logging.info(f"Received candidate with loss {candidate['loss']:.4f}")
            # Add to talent pool if it's good enough:
            if len(self._talent_pool) < self.talent_pool_size or candidate['loss'] < self._talent_pool[-1]['loss']:
                self._talent_pool.append(_move_to_talent_pool(candidate))
                
                self._talent_pool.sort(key=lambda x: x['loss'])  # sort by loss
                if len(self._talent_pool) > self.talent_pool_size:
                    removed = self._talent_pool.pop()  # remove worst
                    _evict_from_pool(removed)
                logging.info(f"Added candidate with loss {candidate['loss']:.4f} to talent pool.")
            else:
                logging.info(f"Candidate with loss {candidate['loss']:.4f} not added to talent pool.")
                _remove_candidate_files(candidate)
            n_proc += 1
        
            
            
    def _trainee_thread_p(self):
        pass   
    
    def _plot_key_callback(self, event):
        if event.key == 'q' or event.key == 'escape':
            logging.info("Quit key pressed, shutting down...")
            self._shutdown.value = True
    
    def run(self):
        self._cand_thread.start()
        self._trainee_thread.start()
        logging.info("Starting UI loop...")
        fig, axes = plt.subplots(2,2) # temp, so we can exit with keypress
        plt.connect('key_press_event', self._plot_key_callback)    
        
        plot_cycles=0
        artists = {'cand_best_img': None, 'cand_hist': None,
                   'trainee_best_img': None, 'trainee_hist': None}

        while not self._shutdown.value:
            # update plots
            if artists['cand_best_img'] is None and len(self._talent_pool) > 0:
                axes[0,0].set_title("Best Candidate so far")
                best_cand = self._talent_pool[0]
                artists['cand_best_img'] = axes[0,0].imshow(best_cand['output_image'])
                axes[0,0].set_xlabel(f"Loss: {best_cand['loss']:.4f}")
            elif artists['cand_best_img'] is not None and len(self._talent_pool) > 0:
                best_cand = self._talent_pool[0]
                artists['cand_best_img'].set_data(best_cand['output_image'])
                best_index = self._talent_pool[0]['n_proc']
                axes[0,0].set_xlabel(f"Index: {best_index}  Loss: {best_cand['loss']:.6f}")

            if artists['cand_hist'] is None and len(self._talent_pool) > 0:
                axes[0,1].set_title("Candidate Loss Histogram")
                artists['cand_hist'] = axes[0,1].hist(self._history['losses'], bins=30, alpha=0.7)
                axes[0,1].set_xlabel("Loss")
                axes[0,1].set_ylabel("Count")
            elif artists['cand_hist'] is not None and len(self._talent_pool) > 0:
                axes[0,1].cla()
                axes[0,1].set_title("Candidate Loss Histogram")
                artists['cand_hist'] = axes[0,1].hist(self._history['losses'], bins=30, alpha=0.7)
                axes[0,1].set_xlabel("Loss")
                axes[0,1].set_ylabel("Count")
                
            plt.draw()
            plt.pause(0.5)
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
    parser.add_argument('-t', '--n_trainee_procs', type=int, default=2, help="Number of processes to use for trainee improvement, if 1 will run in same process.")
    parser.add_argument('-c', '--n_candidates_procs', type=int, default=7, help="Number of processes to use for candidate generation,"+
                                                                                "if 1, will run in same process, separate thread, unless" +
                                                                                "max_candidates > -1 which runs in the main thread before training.")
    
    parser.add_argument('--max_candidates', type=int, default=-1, help="Maximum number of candidates to generate before terminating."+
                                                                       "  = -1 means run indefinitely until stopped (default)." + 
                                                                       "  = 0 means generate no candidates, just load talent pool." +
                                                                       "  > 0 means generate N candidates then stop.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parsed = get_args()
    if parsed.input_image is None and parsed.model_file is None and parsed.test_image is None:
        raise Exception("Need input image (-i) to start training, or model file (-m) to continue.")
    s = GuesserUIDisplay(save_path=parsed.save_path, image_file=parsed.input_image,
                         cand_json=parsed.candidate_params_file, trainee_json=parsed.training_params_file,
                         draft_cycles=parsed.draft_cycles, candidate_luck=parsed.candidate_luck,
                         audition_epochs=parsed.audition_epochs, n_cand_procs=parsed.n_candidates_procs,
                         n_trainee_procs=parsed.n_trainee_procs,)
    # import ipdb; ipdb.set_trace()
    s.run(exit_after=parsed.max_candidates)