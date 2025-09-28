
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

trial_cache_dir = "trial_cache"

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
    def __init__(self, const_params, var_param_names, var_param_values, n_trials, cache_prefix):
        self.const_params = const_params
        self.var_param_names = var_param_names
        self._cache_prefix = cache_prefix
        self.var_param_values = var_param_values
        self.results_raw = None
        self.stats=None
        self.n_trials = n_trials
        self.trials = self._gen_work()
        if not os.path.exists(trial_cache_dir):
            os.makedirs(trial_cache_dir)
        # for rescoring outputs, R, G, and B must be within this fraction (out of  255) of
        self._close_pixel_thresh = 0.05  # the training image to count as "close"
        self._passing_score_thresh = 0.75  # fraction of pixels that must be close to count as passing the test.

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
    
    def run(self, force_recompute=False, n_cpu = 12):
        """
        run on n_cpus in parallel, or if n=1, run serially. (0=all available)
        """
        raw_filename ="%s_tests_results.pkl" % (self._cache_prefix,)
        if force_recompute or not os.path.exists(raw_filename):
            logging.info(f"Raw results file not found:  {raw_filename}, starting new trials....")
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
            logging.info(f"Raw results file found:  {raw_filename}, loading it....")
            # To load raw results:
            with open(raw_filename, "rb") as f:
                self.results_raw = pickle.load(f)
                logging.info(f"Loaded raw results from {raw_filename}")
            
            #self._process_results()
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

    def _rescore_outputs(self, train_images, output_images, title, n_plot=0):
        """
        for each train/output pair:
            Shrink the output iamge down to the train image size.
            return the fraction of pixels whose R, G, and B values are within pct_thresh of the training image.
        """
        def get_close_pixels(train, output):
            # Return fracction o f pixels within threshold, assume images are same size
            diff = np.abs(output.astype(np.int16) - train.astype(np.int16))  # to avoid overflow
            max_variance = self._close_pixel_thresh * 255
            close_mask = (diff[:,:,0] <= max_variance) & (diff[:,:,1] <= max_variance) & (diff[:,:,2] <= max_variance)
            return close_mask


        
        scores = []
        outputs_small = []
        masks = []
        print("Rescoring %i output images..." % len(output_images))
        for train_image, output_image in zip(train_images, output_images):
            if output_image.dtype in [np.float32, np.float64]:
                output_image = np.clip(output_image*255.0, 0, 255).astype(np.uint8)
            if train_image.dtype in [np.float32, np.float64]:
                train_image = np.clip(train_image*255.0, 0, 255).astype(np.uint8)
                
            small_output = cv2.resize(output_image, (train_image.shape[1], train_image.shape[0]), interpolation=cv2.INTER_AREA)
            close_mask = get_close_pixels(train_image, small_output)
            score = np.sum(close_mask) / (close_mask.shape[0] * close_mask.shape[1])
            
            outputs_small.append(small_output)
            scores.append(score)
            masks.append(close_mask)
            
            print("Train mask shape: ", close_mask.shape, "out_small shape: ", small_output.shape, "score: %.4f" % score, 'loss', np.mean((small_output.astype(np.float32)-train_image.astype(np.float32))**2))

        scores = np.array(scores)
        
        if n_plot>0:
            """
            Plot four rows, each pair of 2 rows of n_plot //2 images:
            Top row:  training images
            Bottom row:  output images with pixels within threshold highlighted in green.
            """
            
            n_plot = min(n_plot, len(train_images), len(output_images))
            n_rows = (n_plot//5)
            n_cols = int(np.ceil(n_plot/n_rows))
            fig, axs = plt.subplots(n_rows*2, n_cols, figsize=(15, 6))
            for i in range(n_plot):
                col = i // n_rows
                row = (i % n_rows) * 2
                train_img = train_images[i]
                output_img = outputs_small[i]
                shaded_img = output_img.copy()
                bad_mask = ~masks[i]
                alpha=0.5
                shaded_img[masks[i], :] = alpha* shaded_img[masks[i], :]  + (1-alpha) * np.array(GREEN)
                shaded_img[bad_mask, :] = alpha* shaded_img[bad_mask, :]  + (1-alpha) * np.array(RED)
                collage = self.stack_images([train_img, output_img, shaded_img], sep_space=10)
                axs[row, col].imshow(collage)
                axs[row, col].set_title(f"Score: {scores[i]:.4f}")
                axs[row, col].axis('off')
            # Add spacing between rows
            for ax in axs.flatten():
                ax.axis('off')  
            plt.suptitle(f"Rescoring outputs for {title}\npct_thresh={self._close_pixel_thresh*100:.1f}%", fontsize=16)
            plt.tight_layout()
            fig.subplots_adjust(wspace=0.0, hspace=0.1)
            plt.show()
        return scores

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
                output_images = [r['image'] for r in assigned_results]
                train_images = [r['train_image'] for r in assigned_results]
                assigned_str =", ".join([f"{k}={v}" for k, v in assignments.items()])
                
                new_scores = self._rescore_outputs(train_images, output_images, assigned_str)
                n_trials = len(losses)
                best_ind = np.argmin(losses) if n_trials > 0 else None
                worst_ind = np.argmax(losses) if n_trials > 0 else None 
                median_ind = np.argsort(losses)[n_trials // 2] if n_trials > 0 else None
                return dict(n_trials = len(losses),
                    losses = losses,
                    scores = new_scores,
                    output_images = output_images,
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

    def make_result_collage(self,best,worst,med,scores):
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
        def _add_caption(img, captions, indent = 10,font_scale=1.0, double_strike=True, thickness=2, spacing = 36, line_colors = None):
            """
            Write in the lower-left corner of the image.
            """
            
            captions = [captions] if isinstance(captions, str) else captions
            
            if line_colors is None:
                line_colors = [None] * len(captions)
                
            line_colors = [((0,0,0) if c is None else c) for c in line_colors]
            y = img.shape[0] - indent
            for caption, color in zip(reversed(captions), reversed(line_colors)):
                if double_strike:
                    # Black out the background
                    pad=5
                    (w, h), b = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(img, (indent-pad, y - h-pad), (indent + w + pad, y+pad), (255,255,255), -1)
                    #cv2.putText(img, caption, (indent, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 5, cv2.LINE_AA)
                    cv2.putText(img, caption, (indent, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                else:
                    cv2.putText(img, caption, (indent, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
                y -= spacing
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
            draw_bbox(big_img, bbox, color = GREEN, thickness=2)
            
            if upscale>1.0:
                big_img = cv2.resize(big_img, (0,0), fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
            
            return big_img

        n_passing = np.sum(scores >= self._passing_score_thresh)
        
        best_img = _add_inset(best['image'].copy(), best['train_image'], upscale=2.5, train_upscale=1.5)
        worst_img = _add_inset(worst['image'].copy(), worst['train_image'], upscale=2.5, train_upscale=1.5)
        med_image = _add_inset(med['image'].copy(), med['train_image'], upscale=2.5, train_upscale=1.5)
        stats = 0 * best_img + 255
        stats = _add_caption(stats, ["RGB 'close' thresh:",
                                     "   %.1f%% " % (self._close_pixel_thresh*100,),
                                     "Num close to pass:",
                                     "   %.1f%%" % (self._passing_score_thresh*100,),
                                     "",
                                     "N passing: %i" % (n_passing, )], indent=20, font_scale=1.2,
                                thickness=2,double_strike=False, spacing=42, line_colors = [None, None, None, None, None, RED])

        collage = self.stack_images([_add_caption(med_image, "Median L: %.4f" % med['loss']),
                                     _add_caption(best_img, "Best L: %.4f" % best['loss']),
                                     _add_caption(worst_img, "Worst L: %.4f" % worst['loss']),
                                     stats], sep_space=20)
        
        return collage
    
    def stack_images(self, image_list, sep_space=10):
        blank = np.zeros_like(image_list[0]) + 255
        while len(image_list) <4:
            image_list.append(blank)
        h_sep = np.zeros((image_list[0].shape[0], sep_space, 3), dtype=image_list[0].dtype) + 255
        v_sep = np.zeros((sep_space, image_list[0].shape[1]*2 + sep_space, 3), dtype=image_list[0].dtype) + 255
        collage = np.vstack([
            np.hstack([image_list[0], h_sep, image_list[1]]),
            v_sep,
            np.hstack([image_list[2], h_sep, image_list[3]])
        ])
        return collage

    def plot(self, fig_size=(12, 8)):
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
            grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.2, top=0.85, left=0.152, bottom=0.01, right=0.99)
            fig = plt.figure(figsize=fig_size)
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
                    scores = stat['scores']
                    result_img = self.make_result_collage(best, worst, med, scores)

                    ax.imshow(result_img)
                    ax.set_title(f"mean={stat['mean_loss']:.6f}")
                    ax.axis('off')
                    axes_by_col[-1].append(ax)
                    
                # Write the column headers using annotations above the top row of images
                ax = axes_by_col[-1][0]
                ax.annotate(f"{self.var_param_names[0]} = {val1}", xy=(0.5, 1.2), xycoords='axes fraction', ha='center', fontsize=14)
                ax.axis('off')
                
                # Write the row headers using annotations to the left of the leftmost column of images
                for row, val2 in enumerate(self.var_param_values[1]):
                    ax = axes_by_col[0][row]
                    ax.annotate(f"{self.var_param_names[1]} = {val2}", xy=(-0.25, 0.5), xycoords='axes fraction', ha='center',
                                fontsize=14, rotation=90, rotation_mode='anchor')
                    ax.axis('off')
        

            var_str_1 = get_var_str(self.var_param_values[0])
            var_str_2 = get_var_str(self.var_param_values[1])
            title = "Test(cols):  %s = %s\nTest(rows): %s = %s\n" % (self.var_param_names[0], var_str_1,
                                                                     self.var_param_names[1], var_str_2)
            #title += f"each experiment over {self.n_trials} trials, green insets show training image."
            plt.suptitle(title, fontsize=16)
            

        elif len(self.var_param_names) == 1:
            
            # Create a figure with subplots
            n_cols=3
            if len(self.var_param_values[0]) > n_cols:
                n_rows = np.ceil(len(self.var_param_values[0])/n_cols).astype(int)
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
                scores = self.stats[col]['scores']
                result_img = self.make_result_collage(best, worst, med, scores)

                axs[col].imshow(result_img)

                best_loss = self.stats[col]['losses'][self.stats[col]['best_ind']]
                axs[col].set_title(f"{self.var_param_names[0]} = {param_value}:  "
                                    f"mean loss = {self.stats[col]['mean_loss']:.5f}", fontsize=12)
                axs[col].axis('off')
            # turn off unused axes
            for ax in axs[len(self.var_param_values[0]):]:
                ax.axis('off')

            plt.suptitle("Test:  %s = %s" % (self.var_param_names[0], get_var_str(self.var_param_values[0])) +
                         f"\neach experiment over {self.n_trials} trials, green insets show training image.\n"
                         , fontsize=17)
            plt.tight_layout()
            
        return fig      
    
    def plot_summary(self, fig_size=(12, 5)):
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
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_size,gridspec_kw={'width_ratios': [1, 5]})
        
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
    
    
def get_var_str(values):

    if isinstance(values[0], float):
        par_val_str = {", ".join(["%.3f" % v for v in values])}
    elif isinstance(values[0], int):
        par_val_str = {", ".join([str(v) for v in values])}
    elif isinstance(values[0], str):
        par_val_str = {", ".join(values)}

    return par_val_str


def params_to_cache_filename(params, test_image, trial_ind):
    """
    Create a short string representing the parameters, for use in filenames and caching results.
    [test_name]_[param_string]_[trial_ind].pkl
    """
    param_abbrevs = {
                    'batch_size': 'z%i',
                    'center_weight_params': 'w%.2f_%.3f_%.3f_%.3f_%.3f',  # max_wt, spread, flat, x, y
                    'learning_rate': 'r%5f',
                    'learning_rate_final': 'rf%5f',
                    'epochs_per_cycle': 'e%i',
                    'run_cycles': 'k%i',
                    'grad_sharpness': 'g%.1f',
                    'n_hidden': 'n%i',
                    'n_structure': 't%i',                
        }
    def fill(name, value):
            if value is None:
                return None
            return param_abbrevs[name] % value
        
    if params['n_div']['linear'] > 0:
        structure = "L=%i_%i-param" % (params['n_div']['linear'], params['line_params']) 
    elif params['n_div']['circular'] > 0:
        structure = "C=%i" % (params['n_div']['circular'],)
    else:
        raise ValueError("Must have either linear or circular divisions for simple_experiments.")
    test_string = "%s_%s" % (test_image, structure)
    param_string = "_".join([fill(param_name, param_value) for param_name, param_value in params.items() if param_name in param_abbrevs and
                             fill(param_name, param_value) is not None])
    n_trials_string = "trial=%i" % (trial_ind,)
    
    return f"{test_string}__{param_string}__{n_trials_string}.pkl"


def run_trial(test_params, trial_ind, const_params):
    print(f"\n\n\nRunning trial {trial_ind} with test params: {test_params}\n\n\n")
    params = const_params.copy()
    params.update(test_params)
    # Simulate some processing
    test_image = params['_test_image']
    
    
    del params['_image_file']
    del params['_test_image']
    del params['_model_file']
    params['nogui'] = True
    
    params_cache_file = os.path.join(trial_cache_dir, params_to_cache_filename(params, test_image, trial_ind))
    if not os.path.exists(params_cache_file):
        print(f"Cache file not found: {params_cache_file}, running trial...")
    
        uid = UIDisplay(image_file=None, model_file=None, synth_image_name=test_image, verbose=VERBOSE, **params)
        loss, image = uid.run()
        train_image = uid.get_train_image()
        
        with open(params_cache_file, "wb") as f:
            pickle.dump( (loss, image, train_image, params), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved trial results to {params_cache_file}")
        
    else:
        print(f"Cache file found: {params_cache_file}, loading trial...")
        with open(params_cache_file, "rb") as f:
            data = pickle.load(f)
            if len(data) == 4:
                loss, image, train_image, params = data
            else:
                loss, image, train_image = data
                # Assume params are ok...
            print(f"Loaded trial results from {params_cache_file}")

    return {'test_params': test_params, 'trial_ind': trial_ind, 'loss': loss, 'image': image, 'train_image': train_image, 'params': params}


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