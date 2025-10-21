"""

Plot the loss history for multiple experiments on the same graph for comparison.
Show the final loss in the legend, the best result in bold text.

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json 
import argparse
import os
import logging
from compare_hist import LossComparison
import glob
class MultiLossComparison(LossComparison):

    def __init__(self, dir_pattern, title="Loss Comparison", max_cycles=None, **kwargs):
        """
        
        Plot end-of-cycle loss for all experiments loaded.
        Optionally plot learning rate, annealing noise (sd), and epoch losses as well.
        
        dir_pattern: glob-style pattern for frame directories to look for metadata in.  Will be sorted alphabetically.
        title: Title for the plot
        kwargs: options for plotting (bool, default false)
            * learning_rate: plot learning rate history
            * annealing_noise: plot annealing noise history
            
        """
        self.max_cycles = max_cycles
        self.dir_pattern = dir_pattern
        self.options= {'learning_rate': False,
                       'annealing_noise': False} 
        self.options.update(kwargs)
        self.frame_dirs = sorted( glob.glob(dir_pattern) )
        logging.info(f"Found {len(self.frame_dirs)} frame directories matching pattern {dir_pattern}")
        self.title = title
        
        self.labels = []
        self.hist = self.load_histories()
        
    def plot(self):
        """
        2 plots, shared x axis, learning rates and epoch mean losses.
        x-axis is cycles, different sized cycles will be squished/stretched to fit into a unit space for 1 cycle.
        """
        n_plots = 1 + int(self.options['learning_rate']) + int(self.options['annealing_noise'])
        height_ratios = [1]* (n_plots -1) + [2]

        grid = gridspec.GridSpec(nrows=n_plots, ncols=1, height_ratios=height_ratios)
        fig = plt.figure(figsize=(10,6))
        # Add loss axis first, at the bottom
        
        loss_ax = fig.add_subplot(grid[-1])
        loss_ax.set_title("Loss History (1 point per cycle)")
        
        if self.options['learning_rate']:
            lrate_ax = fig.add_subplot(grid[0], sharex=loss_ax)
            lrate_ax.set_ylabel("Learning Rate")
            lrate_ax.set_yscale('log')
            
            if self.options['annealing_noise']:
                anneal_ax = fig.add_subplot(grid[1], sharex=loss_ax)
                anneal_ax.set_ylabel("Annealing Noise SD")
            else:
                anneal_ax = None
        else:
            if self.options['annealing_noise']:
                anneal_ax = fig.add_subplot(grid[0], sharex=loss_ax)
                anneal_ax.set_ylabel("Annealing Noise SD")
                lrate_ax = None
            else:
                anneal_ax = None
                lrate_ax = None
        
        best_loss = np.inf
        best_label = None
        best_line = None
        artists = {}  # same colors in all plots
        
        for label, meta in self.hist.items():
            print("PLOTTING", label)
            n_cycles = len(meta['loss_history']) if self.max_cycles is None else min(len(meta['loss_history']), self.max_cycles)

            loss_label = "%s (%i cycles)" % (label, n_cycles)
            self.labels.append(label)

            learning_rates =  np.array(meta['learning_rate_history'][:n_cycles]) if self.options['learning_rate'] else None
            annealing_noise = meta['anneal_history'][:n_cycles] if self.options['annealing_noise'] else None  # can be different lengths
            final_loss =  np.array([epoch['final_loss'] for epoch in meta['loss_history'][:n_cycles]])
            n_cycles = len(final_loss)
            cycle_x = np.arange(n_cycles) #/ total_epochs
            lr_y = np.array(learning_rates)
            # loss_y = np.array()
            
            # plot loss
            line, = loss_ax.plot(cycle_x , final_loss, linestyle='-', label=loss_label)
            artists[label] = line
            
            if self.options['learning_rate'] and lrate_ax is not None:
                # plot learning rates
                learn_label = "Learning Rate"
                line, = lrate_ax.step(cycle_x, lr_y, where='post', label=learn_label, color=line.get_color())
                
            if self.options['annealing_noise'] and anneal_ax is not None:
                anneal_x = [np.linspace(i, i+1, len(annealing_noise[i])) for i in range(n_cycles)]
                anneal_x = np.concatenate(anneal_x)
                anneal_y = np.concatenate(annealing_noise)
                # plot annealing noise
                line, = anneal_ax.plot(anneal_x, anneal_y, linestyle='-', label="Annealing Noise SD", color=line.get_color())
                
            if final_loss[-1] < best_loss:
                best_loss = final_loss[-1]
                best_label = label
                best_line = line
                

            
        if best_line is not None:
            # best_line.set_linewidth(3.0)
            best_line.set_zorder(10)
            best_line.set_label("%s (%.6f)  (best)" % (best_label, best_loss))
            logging.info("Best loss: %s with %.6f" % (best_label, best_loss))
        if lrate_ax is not None:            
            lrate_ax.set_title("Learning Rate History")
            lrate_ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            # plt.setp(lrate_ax.get_xticklabels(), visible=False)

        if anneal_ax is not None:
            anneal_ax.set_title("Annealing Noise History")
            anneal_ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        loss_ax.set_title("End of Cycle Losses (full image)")
        # Set legend on loss axis
        loss_ax.legend(artists.values(), artists.keys())
        loss_ax.set_yscale('log')
        loss_ax.set_xlabel("Cycle")
        loss_ax.set_ylabel("Loss")
        loss_ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Make loss x ticks integers only
        loss_ax.xaxis.get_major_locator().set_params(integer=True)
        
        # Turn off x ticks, labels, tick labels
        #epoch_loss_ax.tick_params(labelbottom=False)
        #epoch_loss_ax.set_xticklabels([])
        #epoch_loss_ax.set_xticks([])
        
        # Grids

        plt.subplots_adjust(hspace=0.05)
        plt.tight_layout()
        
        


def get_args():
    parser = argparse.ArgumentParser(description="Compare loss histories from an experiment sequence (i.e. frame dirs indexed by an integer)")
    parser.add_argument('pattern', type=str,
                        help='glob-style pattern for frame directories to look for metadata in.  Will be sorted alphabetically.')
    parser.add_argument('--title', type=str, default="Loss Comparison", help="Title for the plot")
    parser.add_argument('-l','--learning_rate', action='store_true', help="Plot the learning rate history for loaded metadata.")
    parser.add_argument('-a','--annealing_noise', action='store_true', help="Plot the annealing noise history for loaded metadata.")
    parser.add_argument('-m', '--max_cycles', type=int, default=None, help="Maximum number of cycles to plot")

    parsed = parser.parse_args()
    dirs = parsed.pattern.format('*')
    args = {'dir_pattern': dirs,
            'title': parsed.title, 
            'max_cycles': parsed.max_cycles,
            'learning_rate': parsed.learning_rate, 
            'annealing_noise': parsed.annealing_noise}
    return args
    
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    lc = MultiLossComparison(**args)
    lc.plot()
    plt.show()
