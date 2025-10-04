
from image_learn import UIDisplay
import numpy as np
import argparse
import logging
from multiprocessing import Pool, cpu_count
import os
import pickle
import os
import matplotlib.pyplot as plt

from util import draw_bbox
import matplotlib.gridspec as gridspec
import cv2
import json
GREEN = (0, 255, 0)
RED = (255, 0, 0)
VERBOSE=False

from experiment_simple import GREEN, RED, params_to_cache_filename, get_var_str

trial_cache_dir = "trial_cache_inits"

class Experiment(object):
    """
    1. Initialize N random networks. 
    2. Train them all for a small number of epochs.
    3. Take the B best and B median performers:   Is there a significant improvment when using the best vs the median?
    
    """
    def __init__(self, params, n_trials=100, cache_prefix=None, n_subset = 5):
        self.params = params
        self.n_trials = n_trials
        self.n_subset = n_subset
        self._cache_prefix = cache_prefix
        
        self._init_trials_work = self._gen_init_work()
        if not os.path.exists(trial_cache_dir):
            os.makedirs(trial_cache_dir)
        # for rescoring outputs, R, G, and B must be within this fraction (out of  255) of
        self._close_pixel_thresh = 0.05  # the training image to count as "close"
        # Return proportion of pixels in this range as the "score" for step 3.
        
        self.init_results_raw = None
        self.train_results_raw = None
        self.init_stats = None
        self.train_stats =None
        
    def _gen_init_work(self):
        """
        A work unit is: parameters dict, 1 trial index.
        """
        work = []
        for trial in range(self.n_trials):
            work.append( (self.params, trial) )
        return work
    
    def run_init(self, force_recompute=False, n_cpu = 12):
        """
        run on n_cpus in parallel, or if n=1, run serially. (0=all available)
        """
        raw_filename ="%s_tests_results.pkl" % (self._cache_prefix,)
        if force_recompute or not os.path.exists(raw_filename):
            logging.info(f"Raw results file not found:  {raw_filename}")
            if n_cpu == 1:  
                logging.info(f"Running {len(self._init_trials_work)} trials single-threaded...")
                self.results_raw = [run_init_trial(pars, trial_ind) for pars, trial_ind in self._init_trials_work]
            else:
                n_cpu = cpu_count() if n_cpu == 0 else n_cpu
                logging.info(f"Running {len(self._init_trials_work)} trials on {n_cpu} CPUs...")
                with Pool(processes=n_cpu) as pool:
                    self.results_raw = pool.starmap(run_init_trial, self._init_trials_work)
            logging.info("Done running trials... saving results...")
            # To save raw results:
            with open(raw_filename, "wb") as f:
                pickle.dump(self.results_raw, f)
                logging.info(f"Saved raw results to {raw_filename}")
        else:
            logging.info(f"Raw results file found:  {raw_filename}, loading it....")
            # To load raw results:
            with open(raw_filename, "rb") as f:
                self.results_raw = pickle.load(f)
                logging.info(f"Loaded raw results from {raw_filename}, contained %i in")
            self._process_results()
        

    def _process_results(self):
        """
        Each raw init_result is of the form:
            data = {'loss': loss, 
                    'model': model_file_path, 
                    'params': params,
                    'output_image': output_image, 
                    'train_image': train_image, 
                    'score': score}
        Find the best and median subsets, by losses and by score.  (we will plot both.)
        """
        #losses = np.array([r['loss'] for r in self.results_raw])
        scores = np.array([r['score'] for r in self.results_raw])
        order = np.argsort(scores)
        self.init_stats = None
        if self.results_raw is not None:
            # Find best and median for each trial
            self.init_stats = self._get_best_median(self.results_raw, 'init')
            self.train_stats = self._get_best_median(self.results_raw, 'train')

        """
        Plot a summary of all experiments, this is the plot to make the recommendation clear, if the experiment is to have 
        revealed anything at all.
        
        NOTE:  Not implemented for more than 1 experiment parameter.
        
        TITLE:   Data / network:  [test-img],  [circle/line] units:  [n],  color layers: [[n_structure] n_color]
                 Training:  %i cycles, learning rates [%i down to %i], batch size %i, %i epochs/cycle.
                 Exp param:  [param name] = {param values}
                 Trials:  %i trials per parameter set.
        left-plot: example training image 
        right-plot:  Bar graph, number of passing trials for each parameter value.
               Below that, a box plot of the losses for each parameter value.
        :param fig_size:  (width, height) in inches
        :return: figure
        """
        if len(self.var_param_names) > 1:
            raise NotImplementedError("Currently only implemented for 1 varying parameter")
        
        var_param_name = self.var_param_names[0]
        var_param_values = self.var_param_values[0]
        
        n_cols = len(var_param_values)
        fig, axs = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [1, 5]})
        
        # training image:
        axs[0].imshow(self.stats[0]['train_images'][0])
        axs[0].set_title("Example training image\nTest name: %s" % (self.const_params['_test_image'],), fontsize=14)
        axs[0].axis('off')
        
        
        # Bar graph of number of passing trials
        n_passing = [np.sum(stat['scores'] >= self._passing_score_thresh) for stat in self.stats]
        axs[1].bar(range(n_cols), n_passing, color='skyblue')
        axs[1].set_xticks(range(n_cols))
        axs[1].set_xticklabels([str(v) for v in var_param_values])
        axs[1].set_ylabel("Number of Passing Trials", fontsize=14)
        axs[1].set_xlabel(f"{var_param_name}", fontsize=18)
        axs[1].set_ylim(0, self.n_trials)
        axs[1].set_title(f"Passing criterion:  at least {self._passing_score_thresh*100:.1f}% of pixels within {self._close_pixel_thresh*100:.1f}% of training image's RGB values.")
        data_line = f"Test case:  {self.const_params['_test_image']}"
        units_str = "Circle-units:  %i" % (self.const_params['n_div']['circular'],) if self.const_params['n_div']['circular'] > 0 else \
                "Line-units:  %i  (%i param.)" % (self.const_params['n_div']['linear'], self.const_params['line_params'])
        arch_str = f"Structure-units:  {self.const_params['n_structure']},  Color-units:  {self.const_params['n_hidden']}"
        network_line = f"Network:  {units_str},  {arch_str}"
        title = f"{data_line}\n{network_line}\n"
        title += f"Training: batch size {self.const_params['batch_size']}, annealing {self.const_params['run_cycles']} cycles of {self.const_params['epochs_per_cycle']} epochs,"
        title += f"learning-rate init={self.const_params['learning_rate']}, final={self.const_params['learning_rate_final']}\n"

        title += f"Experimental param:  {var_param_name} = {get_var_str(var_param_values)}\n"
        title += f"Num Trials:  {self.n_trials} per parameter value."
        # make more room between axes
        fig.subplots_adjust(wspace=0.4)
        # move top of the bar graph down to make room for title
        box = axs[1].get_position() # get the original position
        axs[1].set_position([box.x0, box.y0, box.width, box.height * 0.7])
        
        plt.suptitle(title, fontsize=14,x=0.05, horizontalalignment='left')
        return fig

def _get_close_pixels(thresh, train, output):
    # Return fracction o f pixels within threshold, assume images are same size
    diff = np.abs(output.astype(np.int16) - train.astype(np.int16))  # to avoid overflow
    max_variance = thresh * 255
    close_mask = (diff[:,:,0] <= max_variance) & (diff[:,:,1] <= max_variance) & (diff[:,:,2] <= max_variance)
    return close_mask

def _score_result(train_img, out_img, thresh):
    if train_img.shape[1]!= out_img.shape[1] or train_img.shape[0] != out_img.shape[0]:
        out_img = cv2.resize(out_img, (train_img.shape[1], train_img.shape[0]), interpolation=cv2.INTER_AREA)
    close_mask = _get_close_pixels(thresh, train_img, out_img)
    return np.sum(close_mask) / np.prod(close_mask.shape)

def run_init_trial(params, trial_ind):
    print(f"\n\n\nRunning trial {trial_ind}.\n\n\n")
    test_image = params['_test_image']
    del params['_image_file']
    del params['_test_image']
    del params['_model_file']
    params['nogui'] = True
    
    params_cache_file = os.path.join(trial_cache_dir, params_to_cache_filename(params, test_image, trial_ind))
    if not os.path.exists(params_cache_file):
        print(f"Cache file not found: {params_cache_file}, running trial...")
    
        uid = UIDisplay(image_file=None, model_file=None, synth_image_name=test_image, verbose=VERBOSE, **params)
        loss, output_image = uid.run()
        train_image = uid.get_train_image()
        score = _score_result(train_image, output_image, thresh=0.05)
        model_file = params_to_cache_filename(params, test_image, trial_ind).replace(".pkl", ".model")
        model_file_path = os.path.join(trial_cache_dir, model_file)
        
        with open(params_cache_file, "wb") as f:
            data = {'loss': loss, 
                    'model': model_file_path, 
                    'params': params,
                    'output_image': output_image, 
                    'train_image': train_image, 
                    'score': score}
            
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved trial results to {params_cache_file}")
        
    else:
        print(f"Cache file found: {params_cache_file}, loading trial...")
        with open(params_cache_file, "rb") as f:
            data = pickle.load(f)
            print(f"Loaded trial results from {params_cache_file}")

    return data


COMMON_PARAMS_stochastic = {'_image_file': None,
              '_model_file': None,
              'batch_size': 8,
              'border': 0.0,
              'center_weight_params': None,
              'display_multiplier': 5.0,
              'downscale': 2.0,
              'frame_dir': None,
              'just_image': None,
              'learning_rate': 1.0,
              'learning_rate_final': 0.001,
              'epochs_per_cycle': 100,
              'run_cycles': 5,
              'sharpness': 1000.0,
              'n_hidden': 5,
              'n_structure': 5,
              'nogui': True,
              
    }

def run_experiment_circles(test_image, n_circles, n_trials = 15, save_fig=True):
    params = COMMON_PARAMS_stochastic.copy()   
    
    # Custom for this experiment but constant:
    params['n_div'] = {'circular': n_circles, 'linear': 0, 'sigmoid': 0}
    params['_test_image'] = test_image
    
    # varying parameters:
    var_param_names = ['grad_sharpness' ]
    var_param_values = [[1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]] 
    n_neurons = (n_circles) * (n_circles +1) //2 + n_circles + int(np.ceil(n_circles/2)) 

    params['n_structure'] = n_neurons
    params['n_hidden'] = n_neurons
    print("N_CIRCLES = %i  -->  USING %i STRUCTURE AND HIDDEN NEURONS" % (n_circles, n_neurons))
    
    # DEBUG, for faster running:
    # params['epochs_per_cycle'] = 3
    # params['run_cycles'] = 1
    # params['learning_rate_final'] = 1.0
    # n_trials=5
    # var_param_values = var_param_values[:2]

    exp_name = "test_%s_Circles=%i" % (test_image, n_circles)
    experiment = Experiment(params, var_param_names, var_param_values, n_trials=n_trials, cache_prefix=exp_name)
    experiment.run()
    fig = experiment.plot(fig_size=(9,12))
    
    
    if not save_fig:
        plt.show()
    else:
        # Save the figure
        fig_filename = "%s_results.png" % (exp_name,)
        fig.savefig(fig_filename, dpi=300)
        logging.info(f"Saved figure to {fig_filename}")
        
    fig = experiment.plot_summary()
    if not save_fig:
        plt.show()
    else:
        # Save the figure
        fig_filename = "%s_summary.png" % (exp_name,)
        fig.savefig(fig_filename, dpi=300)
        logging.info(f"Saved figure to {fig_filename}")

# BATCH version:
# COMMON_PARAMS_BATCH = COMMON_PARAMS_stochastic.copy()
# COMMON_PARAMS_BATCH.update({'batch_size': 4096,
#                      'epochs_per_cycle': 1000,
#                      'run_cycles': 4})


def run_experiment_lines(test_image, line_params, n_lines, n_trials = 15, save_fig=True):
    params = COMMON_PARAMS_stochastic.copy()
    
    params['n_div'] = {'circular': 0, 'linear': n_lines, 'sigmoid': 0}
    params['_test_image'] = test_image
    params['line_params'] = line_params
    n_neurons = (n_lines) * (n_lines +1) //2 + n_lines + int(np.ceil(n_lines/2))
    params['n_structure'] = n_neurons
    params['n_hidden'] = n_neurons

    print("N_LINES = %i  ->  USING %i STRUCTURE AND HIDDEN NEURONS" % (n_lines, n_neurons))
    
    
    var_param_names = ['grad_sharpness']
    var_param_values = [[1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]]
    
    # DEBUG, for faster running:
    # params['epochs_per_cycle'] = 100
    # params['run_cycles'] = 2
    # params['learning_rate_final'] = 0.1
    # n_trials=3
    # var_param_values[0] = var_param_values[0][:2]  # Just shorten this one.
    
    exp_name = "test_%s_NP=%i_Lines=%i" % (test_image, line_params, n_lines)
    experiment = Experiment(params, var_param_names, var_param_values, n_trials=n_trials, cache_prefix=exp_name)
    experiment.run()
    
    
    fig = experiment.plot()
    if not save_fig:
        plt.show()
    else:
        # Save the figure
        fig_filename = "%s_results.png" % (exp_name,)
        fig.savefig(fig_filename, dpi=300)
        logging.info(f"Saved figure to {fig_filename}")
        
    fig = experiment.plot_summary()
    if not save_fig:
        plt.show()
    else:
        # Save the figure
        fig_filename = "%s_summary.png" % (exp_name,)
        fig.savefig(fig_filename, dpi=300)
        logging.info(f"Saved figure to {fig_filename}")
        


def run_experiments():
    
    run_experiment_circles(n_circles = 1, test_image = 'circle_center')
    run_experiment_circles(n_circles = 2, test_image = 'circles_2_rand')
    run_experiment_circles(n_circles = 3, test_image = 'circles_3_rand')
    run_experiment_circles(n_circles = 4, test_image = 'circles_4_rand')
    
    run_experiment_lines(line_params=2, n_lines=1, test_image = 'line_rand')
    run_experiment_lines(line_params=2, n_lines=4, test_image = 'lines_4_rand')
    run_experiment_lines(line_params=2, n_lines=2, test_image = 'lines_2_rand')
    run_experiment_lines(line_params=2, n_lines=3, test_image = 'lines_3_rand')

    run_experiment_lines(line_params=3, n_lines=1, test_image = 'line_rand')
    run_experiment_lines(line_params=3, n_lines=2, test_image = 'lines_2_rand')
    run_experiment_lines(line_params=3, n_lines=3, test_image = 'lines_3_rand')
    run_experiment_lines(line_params=3, n_lines=4, test_image = 'lines_4_rand')

    
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_experiments()