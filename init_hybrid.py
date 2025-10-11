"""
Initialize a new network using the divider layer of an existing network (the first layer of weights).
Useful for exploring the effects of different architectures for the rest of the network (structure and color layer sizes).
"""
import logging
import os
import json
import argparse
import pickle
import numpy as np
from image_net import NNetImage



class HybridMaker(object):
    """
    Initialize a new network using the divider layer of an existing network (the first layer of weights), and randomly initializing the rest.
    """
    def __init__(self, source_model_file, hybrid_model_file, n_structure=None, n_color=None, clobber=False):
        self.source_model_file = source_model_file
        self.n_structure = n_structure
        self.n_color = n_color
        
        if not os.path.exists(self.source_model_file):
            raise ValueError("Source model file %s does not exist" % self.source_model_file)
        self.load_source_model_state()
        self.create_hybrid_model()

        self.hybrid_model_file = self._get_out_filename(hybrid_model_file)
        self.clobber = clobber
        if os.path.exists(self.hybrid_model_file) and not self.clobber:
            raise ValueError("Output model file %s already exists, use --clobber to overwrite" % self.hybrid_model_file)


        self.save_hybrid_model()
        
    def load_source_model_state(self):
        with open(self.source_model_file, 'rb') as f:
            self.source_state = pickle.load(f)
        # Check only one kind of divider type in n_div is nonzero:
        if sum([1 for _, n_units in self.source_state['n_div'].items() if n_units > 0]) != 1:
            raise ValueError("Source model has more than one kind of divider layer (%s), not supported." % str(self.source_state['n_div']))
        logging.info("Loaded source model from %s" % self.source_model_file)
        
    def create_hybrid_model(self):
        """
        Create a new model using the new parameters, then copy the existing model's divider layer weights.
        NOTE:  Not implemented for models with more than one kind of divider type (e.g. lines AND circles), so this will always be
        the first matrix in the weight list, self.source_state['weights'][0].

        NNetImage.__init__(self, image, n_hidden, n_structure, n_div, state_file=None, batch_size=64, sharpness=1000.0, grad_sharpness=3.0, 
                 learning_rate_initial=1.0, n_train=0, center_weight_params=None, line_params=3, dry_run=False):
                 
                 
                 

        self._sim = NNetImage(n_div=self.n_div, n_hidden=n_hidden, n_structure=n_structure, learning_rate_initial=self._learn_rate,
                        batch_size=self._batch_size, state_file=state_file, image=self._image, line_params=self._line_params,
                        n_train=self._n_train, center_weight_params=self._center_weight_params, dry_run=dry_run,**kwargs)
                        
                        
        """
        # Construct the new model:
        image = self.source_state['image']
        n_color = self.source_state['n_hidden'] if self.n_color is None else self.n_color  # Called the "hidden" layer in old terminology
        n_structure = self.source_state['n_structure'] if self.n_structure is None else self.n_structure
        n_div = self.source_state['n_div']
        sharpness = self.source_state['sharpness']
        grad_sharpness = self.source_state['grad_sharpness']
        n_train = self.source_state['n_train']
        dry_run = False
        
        self.hybrid_model = NNetImage(image=image, n_hidden=n_color, n_structure=n_structure, n_div=n_div, state_file=None,
                                     sharpness=sharpness, grad_sharpness=grad_sharpness,
                                     n_train=n_train, dry_run=dry_run)
        
        # Now copy the weights
        hybrid_weights = self.hybrid_model._model.get_weights()
        #  divider layer weights (center x, center y) (angle) (sharpness)
        for layer in range(3):
            source_div_weights = self.source_state['weights'][layer]       
            if hybrid_weights[layer].shape != source_div_weights.shape:
                raise ValueError("Source model divider weights shape %s does not match new model divider weights shape %s" % (str(source_div_weights.shape), str(hybrid_weights[layer].shape)))
            hybrid_weights[layer] = source_div_weights
        n_div_units = hybrid_weights[0].shape[0]
        # Structure layer if there is one, then color layer, then output layer:
        layer = 3  # index into weight tensor list
        n_src_div_units = self.source_state['weights'][0].shape[0]
        quiet = 1.0 #/ 10.0
        hybrid_size = [n_div_units, self.n_structure, self.n_color, 3] if self.n_structure > 0 else [n_div_units, self.n_color, 3]
        h_layer = 1
        # import ipdb; ipdb.set_trace()

        while layer < len(hybrid_weights):
            source_wts = self.source_state['weights'][layer]
            source_bias  = self.source_state['weights'][layer+1]
            # First the weight matrix, then the bias vector
            n_src_inputs = source_wts.shape[0]
            n_src_units = source_wts.shape[1]
            n_hyb_units = hybrid_size[h_layer]
            n_hyb_inputs = hybrid_size[h_layer-1]
            
            n_copy_in = min(n_src_inputs, n_hyb_inputs)
            n_copy_out = min(n_src_units, n_hyb_units)
            hybrid_weights[layer] *= quiet  # make all weights small first
            logging.info("Copying weights for layer %i:  source %s to hybrid %s, block shape %i x %i" % (layer, str(source_wts.shape), str(hybrid_weights[layer].shape), n_copy_in, n_copy_out))
            hybrid_weights[layer][:n_copy_in, :n_copy_out] = source_wts[:n_copy_in, :n_copy_out]
            
            
            n_bias_copy = min(len(source_bias), len(hybrid_weights[layer+1]))
            hybrid_weights[layer+1] *= quiet  # make all biases small first
            logging.info("Copying biases for layer %i:  source %s to hybrid %s, block shape %i" % (layer, str(source_bias.shape), str(hybrid_weights[layer+1].shape), n_bias_copy))
            hybrid_weights[layer+1][:n_bias_copy] = source_bias[:n_bias_copy]

            layer+=2
            h_layer+=1
            

        self.hybrid_model._model.set_weights(hybrid_weights)
        logging.info("Created hybrid model with divider layer from %s, structure units: %d, color units: %d" % (self.source_model_file, n_structure, n_color))

    def get_arch_str(self):
        """
        for 15 lines and 15 circles + 10 color units, should look like:
        15c-15l_10h
        """
        arch_str = ""

        for div_type in ['circular', 'linear', 'sigmoid']:
            n_div = self.hybrid_model.n_div.get(div_type, 0)
            if n_div > 0:
                arch_str += "%i%s-" % (n_div, div_type[0])
        arch_str = arch_str[:-1]  # remove trailing -
        if self.hybrid_model.n_structure >0:
            arch_str += "_%it" % (self.hybrid_model.n_structure,)
        arch_str += "_%ic" % (self.hybrid_model.n_hidden,)
        return arch_str
    
    def _get_out_filename(self, hybrid_model_file):
        if hybrid_model_file is None:
            model_filename = os.path.basename(self.source_model_file)
            model_prefix = model_filename[:model_filename.index('_model_')]
            model_filename = "%s_model-hybrid_%s.pkl" % (model_prefix, self.get_arch_str())
            input_path = os.path.split(os.path.abspath(self.source_model_file))[0]
            hybrid_model_file = os.path.join(input_path, model_filename)
        print("\n\nWriting to: %s\n\n" % hybrid_model_file)
        return hybrid_model_file
    
    def save_hybrid_model(self):
        logging.info("Saving hybrid model to %s" % self.hybrid_model_file)  

        self.hybrid_model.save_state(model_filename = self.hybrid_model_file)
        
        
def get_args():
    parser = argparse.ArgumentParser(description="Initialize a new network using the divider layer of an existing network.")
    parser.add_argument('source_model_file', type = str, help = "Path to the source model file (pickle) containing the network weights w/ divider layer  to copy.")
    parser.add_argument('-t', '--n_structure', type=int, default=None, help="Number of structure units (after divider layer) for the new network (default: same as source model).")
    parser.add_argument('-c', '--n_color', type=int, default=None, help="Number of color units (after structure layer, if present) for the new network (default: same as source model).")
    parser.add_argument('-o', '--hybrid_model_file', type=str, default=None, help="Output model file (pickle) for the new network (else same as source model file).")
    parser.add_argument('--clobber', action='store_true', help="Overwrite output model file if it exists.")
    parsed = parser.parse_args()
    kwargs = {'source_model_file': parsed.source_model_file,
              'n_structure': parsed.n_structure,
              'n_color': parsed.n_color,
              'hybrid_model_file': parsed.hybrid_model_file,
              'clobber': parsed.clobber}
    return kwargs

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    h = HybridMaker(**args)