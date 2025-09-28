
from image_learn import UIDisplay
import numpy as np
import argparse
import logging
from multiprocessing import Pool, cpu_count
import os
import pickle
import matplotlib.pyplot as plt
from util import draw_bbox
import matplotlib.gridspec as gridspec
import cv2
NEON_GREEN = (57, 255, 20)

def run_trial(test_params, trial_ind, const_params):

    logging.info(f"\n\n\nRunning trial {trial_ind} with test params: {test_params}\n\n\n")
    params = const_params.copy()
    params.update(test_params)
    # Simulate some processing
    test_image = params['_test_image']
    del params['_image_file']
    del params['_test_image']
    del params['_model_file']
    params['nogui'] = True
    uid = UIDisplay(image_file=None, model_file=None, synth_image_name=test_image, **params)
    loss, image = uid.run()
    return {'test_params': test_params, 'trial_ind': trial_ind, 'loss': loss, 'image': image, 'train_image': uid.get_train_image()}

class Experiment(object):
    """
    Parameter sweep, multiple trials per parameter set.
    
    For each parameter varying, create a list of statistics, one dict for each value, with keys:
       value: the parameter value
       losses: list of final losses for each trial
       best_image: image with lowest loss
       worst_image: image with highest loss
       median_image: image with median loss
       mean_loss: mean of losses
       sd_loss: standard deviation of losses
       n_trials: number of trials
    
    if 2 parameters vary, create a list of lists.  If 3 parameters vary, create a list of lists of lists, etc.
    
    value stored in self.stats.
    
    """
    def __init__(self, const_params, var_param_names, var_param_values, n_trials=10,name=None):
        self.const_params = const_params
        self.var_param_names = var_param_names
        self._name = name
        self.var_param_values = var_param_values
        self.results_raw = None
        self.stats=None
        self.n_trials = n_trials
        self.trials = self._gen_work()
        
    def _gen_work(self):
        """
        A work unit is 1 dict of test parameters, 1 trial index, and the constant parameters dict.
        """
        
        param_combos = [{}]  # start with one empty combo
        for p_ind, param_name in enumerate(self.var_param_names):
            new_combos = []
            for combo_set in param_combos:  # for each existing combo, add all the new ones:
                for param_value in self.var_param_values[p_ind]:
                    new_combo = combo_set.copy()
                    new_combo[param_name] = param_value
                    new_combos.append(new_combo)
            param_combos = new_combos
        work = []
        for combo in param_combos:
            for trial_ind in range(self.n_trials):
                work.append( (combo, trial_ind, self.const_params) )
        logging.info(f"Generated {len(work)} total trials ({len(param_combos)} parameter combinations, {self.n_trials} trials each)")
        return work
    
    def run(self, n_cpu = 10, load=False):
        """
        run on n_cpus in parallel, or if n=1, run serially. (0=all available)
        """
        raw_filename ="%s_tests_results.pkl" % (self._name,) if self._name is not None else "test_results.pkl"
        if not load:
            if n_cpu == 1:  
                self.results_raw = [run_trial(combo, trial_ind, const_pars) for combo, trial_ind, const_pars in self.trials]
            else:
                n_cpu = cpu_count() if n_cpu == 0 else n_cpu
                with Pool(processes=n_cpu) as pool:
                    self.results_raw = pool.starmap(run_trial, self.trials)

            # To save raw results:
            with open(raw_filename, "wb") as f:
                pickle.dump(self.results_raw, f)
                logging.info(f"Saved raw results to {raw_filename}")
        else:
            # To load raw results:
            with open(raw_filename, "rb") as f:
                self.results_raw = pickle.load(f)
                logging.info(f"Loaded raw results from {raw_filename}")
            self._process_results()
                
                
        self._process_results()
        
        
        
    def _process_results_old(self):
        """
        Create the stats structure from the raw results.
        
        """
        if len(self.var_param_names) > 1:
            raise NotImplementedError("Currently only implemented for 1 varying parameter")
        var_param_name = self.var_param_names[0]
        var_param_values = self.var_param_values[0]
        stats = []
        for ind, val in enumerate(var_param_values):
            stat = {'value': val,
                    'losses': [], 
                    'train_image': self.results_raw[0]['train_image'],
                    'n_trials': 0, 
                    'mean_loss': None,
                    'best_image': None, 
                    'worst_image': None, 
                    'median_image': None}
            
            val_results = [r for r in self.results_raw if r['test_params'][var_param_name] == val]
            logging.info("Found %i results for parameter %s = %s" % (len(val_results), var_param_name, str(val))
                         )
            stat['losses'] = np.array([r['loss'] for r in val_results])
            stat['n_trials'] = stat['losses'].size
            if stat['n_trials'] > 0:
                stat['mean_loss'] = float(np.mean(stat['losses']))
                stat['sd_loss'] = float(np.std(stat['losses']))
                best_ind = np.argmin(stat['losses'])
                worst_ind = np.argmax(stat['losses'])
                median_ind = np.argsort(stat['losses'])[len(stat['losses'])//2]
                stat['best_image'] = val_results[best_ind]['image']
                stat['worst_image'] = val_results[worst_ind]['image']
                stat['median_image'] = val_results[median_ind]['image']
                stat['n_trials'] = len(val_results)
                stat['train_image'] = val_results[0]['train_image']
            else:
                stat['mean_loss'] = None
                stat['sd_loss'] = None
            stats.append(stat)
        self.stats = stats
        
    def _process_results(self):
        """
        Create the stats structure from the raw results.
        Recursively, for each variable parameter:
           for each value of that parameter:
               return a list of stats with that parameter fixed and the rest varying
        """
        def process_level(assignments, varying_params):
            if len(varying_params) == 0:
                # base case: return stats for this assignment
                assigned_results = [r for r in self.results_raw if all(r['test_params'].get(k) == v for k, v in assignments.items())]
                losses = np.array([r['loss'] for r in assigned_results])
                train_images = [r['train_image'] for r in assigned_results]
                n_trials = len(losses)
                best_ind = np.argmin(losses) if n_trials > 0 else None
                worst_ind = np.argmax(losses) if n_trials > 0 else None 
                median_ind = np.argsort(losses)[n_trials // 2] if n_trials > 0 else None
                return dict(n_trials = len(losses),
                    losses = losses,
                    train_images = train_images,
                    mean_loss = float(np.mean(losses)),
                    sd_loss = float(np.std(losses)),
                    best_ind = best_ind,
                    worst_ind = worst_ind,
                    median_ind = median_ind,
                    best_image = assigned_results[best_ind]['image'],
                    worst_image = assigned_results[worst_ind]['image'],
                    median_image = assigned_results[median_ind]['image'],
                    )
            else:
                param_name = varying_params[0]
                param_values = self.var_param_values[self.var_param_names.index(param_name)]
                stats_list = []
                for val in param_values:
                    new_assignments = assignments.copy()
                    new_assignments[param_name] = val
                    stats_list.append(process_level(new_assignments, varying_params[1:]))
                return stats_list
        self.stats = process_level({}, self.var_param_names)
        # import pprint
        # pprint.pprint(self.stats)
        
    def make_result_collage(self,best,worst,med):
        """
        Stack the images, 2 by 2, 
        [[ median, best  ]
         [       , worst ]]  # i.e. with a blank space in lower left.


        In the upper right,inset the training image for each

        :param best:  {'output': image,'train': image}
        :param worst: {'output': image,'train': image}
        :param med:   {'output': image,'train': image}
        :return: collage image
        
        """
        def _add_caption(img, caption, indent = 10):
            """
            Write in the lower-left corner of the image.
            """
                
            cv2.putText(img, caption, (indent, img.shape[0] - indent), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,0), 1, cv2.LINE_AA)
            return img
        
        def _add_inset(big_img, small_img, upscale=1.0, train_upscale=1.0):
            
            if small_img.dtype in [np.float32, np.float64]:
                small_img = np.clip(small_img*255.0, 0, 255).astype(np.uint8)
            if big_img.dtype in [np.float32, np.float64]:
                big_img = np.clip(big_img*255.0, 0, 255).astype(np.uint8)
                
            if train_upscale!=1.0:
                small_img = cv2.resize(small_img, (0,0), fx=train_upscale, fy=train_upscale, interpolation=cv2.INTER_NEAREST)

            big_img[2:2+small_img.shape[0], 2:2+small_img.shape[1], :] = small_img
            bbox = {'x':(1, 1+small_img.shape[1]+1), 'y':(1, 1+small_img.shape[0]+1)}
            draw_bbox(big_img, bbox, color = NEON_GREEN, thickness=2)
            
            if upscale>1.0:
                big_img = cv2.resize(big_img, (0,0), fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
            
            return big_img
        
        best_img = _add_inset(best['image'].copy(), best['train_image'], upscale=2.0, train_upscale=1.5)
        worst_img = _add_inset(worst['image'].copy(), worst['train_image'], upscale=2.0, train_upscale=1.5)
        med_image = _add_inset(med['image'].copy(), med['train_image'], upscale=2.0, train_upscale=1.5)
        blank = np.zeros_like(best_img) + 255
        collage = np.vstack([
            np.hstack([_add_caption(med_image, "Median L: %.4f" % med['loss']), _add_caption(best_img, "Best L: %.4f" % best['loss'])]),
            np.hstack([blank, _add_caption(worst_img, "Worst L: %.4f" % worst['loss'])])
        ])
        return collage

    def plot(self):
        """
        Plot the results for a single parameter sweep.
        For the N values, plot N columns:
              * top row: best image
              * middle row: median image
              * bottom row: worst image
              
        In the title of each column (top row), show the parameter value, mean & sd of the losses.
        """
        if len(self.var_param_names) > 2:
            raise NotImplementedError("Currently only implemented for 1 or 2 varying parameters")
        
        if len(self.var_param_names) == 2:
            """
            Make a table.  Along the top show the different values of parameeter 1,
            along the right side show the different values of parameter 2.
            in each cell, show the median image for that combination of parameters, above it print the mean sd of the losses,
            """
            # show the training image in a separate window
            # tr_fig, tr_ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
            # tr_ax.imshow(self.stats[0][0]['train_image'])
            # tr_ax.set_title("Training Image")
            # tr_ax.axis('off')
            n_rows = len(self.var_param_values[1]) 
            n_cols = len(self.var_param_values[0]) 
            # make room at top and on left for labels
            grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.4, top=0.85, left=0.15)
            fig = plt.figure()
            axes_by_col = []
            for col, val1 in enumerate(self.var_param_values[0]):
                axes_by_col.append([])
                for row, val2 in enumerate(self.var_param_values[1]):
                    ax = fig.add_subplot(grid[row, col])
                    stat = self.stats[col][row]
                    best_ind = stat['best_ind']
                    worst_ind = stat['worst_ind']
                    median_ind = stat['median_ind']
                    
                    best= {'image':stat['best_image'],
                           'train_image':stat['train_images'][best_ind],
                            'loss':stat['losses'][best_ind]}
                    worst = {'image':stat['worst_image'],
                             'train_image':stat['train_images'][worst_ind],
                             'loss':stat['losses'][worst_ind]}
                    med = {'image':stat['median_image'],
                           'train_image':stat['train_images'][median_ind],
                           'loss':stat['losses'][median_ind]}
                    result_img = self.make_result_collage(best, worst, med)

                    ax.imshow(result_img)
                    ax.set_title(f"mean={stat['mean_loss']:.6f}")
                    ax.axis('off')
                    axes_by_col[-1].append(ax)
                    
                # Write the column headers using annotations above the top row of images
                ax = axes_by_col[-1][0]
                ax.annotate(f"{self.var_param_names[0]} = {val1}", xy=(0.5, 1.2), xycoords='axes fraction', ha='center', fontsize=12)
                ax.axis('off')
                
                # Write the row headers using annotations to the left of the leftmost column of images
                for row, val2 in enumerate(self.var_param_values[1]):
                    ax = axes_by_col[0][row]
                    ax.annotate(f"{self.var_param_names[1]} = {val2}", xy=(-0.3, 0.5), xycoords='axes fraction', ha='center',
                                fontsize=12, rotation=90, rotation_mode='anchor')
                    ax.axis('off')
        

            var_str_1 = get_var_str(self.var_param_values[0])
            var_str_2 = get_var_str(self.var_param_values[1])
            title = "Test(cols):  %s = %s\nTest(rows): %s = %s\n" % (self.var_param_names[0], var_str_1,
                                                                     self.var_param_names[1], var_str_2)
            #title += f"each experiment over {self.n_trials} trials, green insets show training image."
            plt.suptitle(title, fontsize=16)
        elif len(self.var_param_names) == 1:
            
            # Create a figure with subplots
            if len(self.var_param_values[0]) > 3:
                n_rows = np.ceil(len(self.var_param_values[0])/3).astype(int)
                n_cols = np.ceil(len(self.var_param_values[0])/n_rows).astype(int)
                fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(13, 8.5))
                axs = axs.flatten()
            else:
                fig, axs = plt.subplots(nrows=1, ncols=len(self.var_param_values[0]), figsize=(13, 8.5))

            # Turn off the rest of the axes in the first column
            for row in range(1, 3):
                axs[ 0].axis('off')

            for col, param_value in enumerate(self.var_param_values[0]):
                # Plot best images
                best_ind = self.stats[col]['best_ind']
                worst_ind = self.stats[col]['worst_ind']
                median_ind = self.stats[col]['median_ind']
                best= {'image':self.stats[col]['best_image'],
                       'train_image':self.stats[col]['train_images'][best_ind],
                       'loss':self.stats[col]['losses'][best_ind]}
                worst = {'image':self.stats[col]['worst_image'],
                         'train_image':self.stats[col]['train_images'][worst_ind],
                         'loss':self.stats[col]['losses'][worst_ind]}
                med = {'image':self.stats[col]['median_image'],
                       'train_image':self.stats[col]['train_images'][median_ind],
                       'loss':self.stats[col]['losses'][median_ind]}

                result_img = self.make_result_collage(best, worst, med)

                axs[col].imshow(result_img)

                best_loss = self.stats[col]['losses'][self.stats[col]['best_ind']]
                axs[col].set_title(f"{self.var_param_names[0]} = {param_value}\n"
                                    f"  mean loss = {self.stats[col]['mean_loss']:.5f}")
                axs[col].axis('off')
            # turn off unused axes
            for ax in axs[len(self.var_param_values[0]):]:
                ax.axis('off')

            plt.suptitle("Test:  %s = %s" % (self.var_param_names[0], get_var_str(self.var_param_values[0])) +
                         f"\neach experiment over {self.n_trials} trials, green insets show training image.\n"
                         , fontsize=17)
            plt.tight_layout()
            
            
def get_var_str(values):

    if isinstance(values[0], float):
        par_val_str = {", ".join(["%.3f" % v for v in values])}
    elif isinstance(values[0], int):
        par_val_str = {", ".join([str(v) for v in values])}
    elif isinstance(values[0], str):
        par_val_str = {", ".join(values)}

    return par_val_str

COMMON_PARAMS = {'_image_file': None,
              '_model_file': None,
              'batch_size': 4,
              'border': 0.0,
              'center_weight_params': None,
              'display_multiplier': 5.0,
              'downscale': 2.0,
              'frame_dir': None,
              'just_image': None,
              'learning_rate': 1.0,
              'learning_rate_final': 0.001,
              'epochs_per_cycle': 100,
              'run_cycles': 4,
              'sharpness': 1000.0,
              'n_hidden': 5,
              'n_structure': 5,
              'nogui': True,
              
    }

def run_experiment_circles(n_cpu = 14, n_trials = 10):
    params = COMMON_PARAMS.copy()   
    
    # Custom for this experiment but constant:
    params['n_div'] = {'circular': 2, 'linear': 0, 'sigmoid': 0}
    params['_test_image'] = 'circles_2_rand'
    
    # varying parameters:
    var_param_names = ['grad_sharpness' ]
    var_param_values = [[1.0, 2.0, 3.0, 4.0, 5.0, 10.0]]
    
    # DEBUG, for faster running:
    # params['epochs_per_cycle'] = 25
    # params['run_cycles'] = 1
    # params['learning_rate_final'] = 1.0
    # n_trials=5
    # var_param_values = var_param_values[:5]
    load = False  # False to recalculate debug data, True to load it
    
    experiment = Experiment(params, var_param_names, var_param_values, n_trials=n_trials, name = "circle_grad_sharpness_test")
    experiment.run(n_cpu=n_cpu,load=load)
    print("\n\n\n\nExperiment results summary:")
    experiment.plot()   
    plt.show()
    
def run_experiment_lines(n_cpu = 14, n_trials = 10):
    params = COMMON_PARAMS.copy()
    
    params['n_div'] = {'circular': 0, 'linear': 2, 'sigmoid': 0}
    params['_test_image'] = 'lines_2_rand'
    
    var_param_names = ['grad_sharpness', 'line_params']
    var_param_values = [[1.0, 2.0, 3.0, 5.0, 10.0], [ 2, 3]]  #  
    
    # DEBUG, for faster running:
    # params['epochs_per_cycle'] = 10
    # params['run_cycles'] = 2
    # params['learning_rate_final'] = 0.1
    # n_trials=3
    # var_param_values[0] = var_param_values[0][:2]  # Just shorten this one.
    
    load=False  # False to recalculate debug data, True to load it
    
    experiment = Experiment(params, var_param_names, var_param_values, n_trials=n_trials, name = "line_params_2_or_3_test")
    experiment.run(n_cpu=n_cpu,load=load)        
    experiment.plot()

    plt.show()
    
    
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_experiment_circles()
    run_experiment_lines()