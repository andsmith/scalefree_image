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


class LossComparison(object):
    
    def __init__(self, frame_dirs, title="Loss Comparison"):
        self.frame_dirs = frame_dirs
        self.title = title
        
        self.labels = []
        self.hist = self.load_histories()
        
    def load_histories(self):
        hist = {}
        for d in self.frame_dirs:
            meta_path = [f for f in os.listdir(d) if 'metadata' in f and f.endswith('.json')]
            if len(meta_path) == 0  :
                logging.warning("No metadata.json found in %s, skipping" % d)
                continue
            meta_path = os.path.join(d, meta_path[0])
            if not os.path.exists(meta_path):
                raise ValueError("Metadata file %s does not exist" % meta_path)
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if 'loss_history' not in meta:
                logging.warning("No loss_history found in %s, skipping" % d)
                continue
            hist[d] = meta
            logging.info("Loaded history from %s with %d entries" % (d, len(meta['loss_history'])))
            # # show top level keys
            # for k in meta.keys():
            #     logging.info("  %s: %s" % (k, str(meta[k])[:100]))

            # # import ipdb; ipdb.set_trace()

            hist[d]['epoch_mean_losses'] = [[np.mean(epoch_minibatch_losses)  for epoch_minibatch_losses in cycle['epochs']] for cycle in meta['loss_history']]
        return hist
    
    def plot(self):
        """
        2 plots, shared x axis, learning rates and epoch mean losses.
        x-axis is cycles, different sized cycles will be squished/stretched to fit into a unit space for 1 cycle.
        """
        grid = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[1,2,2])
        fig = plt.figure(figsize=(10,6))
        lrate_ax = fig.add_subplot(grid[0])
        loss_ax = fig.add_subplot(grid[1], sharex=lrate_ax)
        epoch_loss_ax = fig.add_subplot(grid[2], sharex=lrate_ax)
    
        
        epoch_loss_ax.set_yscale('log')
        epoch_loss_ax.set_ylabel("Epoch Losses (32768 samples/epoch)")
        
        lrate_ax.set_ylabel("Learning Rate")
        lrate_ax.set_yscale('log')
        # epoch_loss_ax.set_xlabel("(time normalized over all cycles)")
        
        best_loss = np.inf
        best_label = None
        best_line = None
        artists = {}  # same colors in both plots
        
        for d, meta in self.hist.items():
            
            print("PLOTTING", d)
            label = d[2:-1]
            loss_label = "%s (%i cycles)" % (label, len(meta['epoch_mean_losses']))
            self.labels.append(label)
            epoch_mean_losses = meta['epoch_mean_losses']
            learning_rates = meta['learning_rate_history']
            total_epochs = len(epoch_mean_losses)

            cycle_x = np.arange(total_epochs) #/ total_epochs
            lr_y = np.array(learning_rates)
            # loss_y = np.array()
            
            # plot learning rates
            line, = lrate_ax.step(cycle_x, lr_y, where='post', label=loss_label)
            artists[label] = line
            
            # Plot end of cycle losses:
            eoc_loss_x = cycle_x
            eoc_loss_y = np.array([cycle['final_loss'] for cycle in meta['loss_history']])
            loss_ax.plot(eoc_loss_x, eoc_loss_y,  linestyle='-', color=line.get_color(), label=loss_label)
            
            
            # plot losses
            epoch_loss_x = []
            epoch_loss_y = []
            for cyc_i, cycle in enumerate(epoch_mean_losses):
                eml_x = np.linspace(0, 1, len(cycle), endpoint=False) + cyc_i
                epoch_loss_x.extend(eml_x)
                epoch_loss_y.extend(cycle)
            epoch_loss_x = np.array(epoch_loss_x) 
            epoch_loss_y = np.array(epoch_loss_y)
            loss_line = epoch_loss_ax.plot(epoch_loss_x , epoch_loss_y, color=line.get_color())[0]
            
            final_loss = epoch_loss_y[-1]
            if final_loss < best_loss:
                best_loss = final_loss
                best_label = label
                best_line = loss_line
                
            logging.info("  %s: final loss %.6f" % (label, final_loss))            # show final loss in legend
            loss_line.set_label("%s (%.6f)" % (label, final_loss))
            
        if best_line is not None:
            # best_line.set_linewidth(3.0)
            best_line.set_zorder(10)
            best_line.set_label("%s (%.6f)  (best)" % (best_label, best_loss))
            logging.info("Best loss: %s with %.6f" % (best_label, best_loss))
            
        epoch_loss_ax.legend()
        lrate_ax.legend(artists.values(), artists.keys())
        lrate_ax.set_title("Learning Rate History")
        epoch_loss_ax.set_title("Loss History (1 point per epoch)")
        loss_ax.set_title("End of Cycle Losses (full image)")
        # Turn off x ticks, labels, tick labels
        #epoch_loss_ax.tick_params(labelbottom=False)
        #epoch_loss_ax.set_xticklabels([])
        #epoch_loss_ax.set_xticks([])
        # Grids
        epoch_loss_ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        lrate_ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        loss_ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.setp(lrate_ax.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=0.05)
        plt.tight_layout()
        
        


def get_args():
    parser = argparse.ArgumentParser(description="Compare loss histories from multiple experiments")
    parser.add_argument('frame_dirs', type=str, nargs='+',
                        help='Directories containing *metadata*.json files with loss histories')
    parser.add_argument('--title', type=str, default="Loss Comparison", help="Title for the plot")
    parsed = parser.parse_args()
    dirs = parsed.frame_dirs
    args = {'frame_dirs': dirs, 'title': parsed.title}
    return args
    
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    lc = LossComparison(**args)
    lc.plot()
    plt.show()
