"""
TK app to find lines in an image to seed divider units.
Adjust params until they look right, then save, load w/image_learn.py with the --line_init_file option.
Also, optionally use the --freeze_dividers option to keep them fixed during training (and just learn colors).


Algorithm:
   1.  Preprocessing (dowsampling, Gaussian blur)
   2.  Canny edge detection (find edges between regions w/contrasting colors)
   3.  Hough line detection (find straight lines among the edges)
   4.  Divider initialization:  for each Hough line, determine its support 
       (how many edge pixels are within its band) and select the top N lines by support
       to initialize divider lines.


Params to adjust in app:  default_value [range_low, range_high]

    [Sliders numbered Ps1, Ps2, ... for the GUI.]

    Preprocessing:
    Ps1- pyramid down levels:  1 [0, 10] 
    Ps2- Gaussian blur kernel size: 5 [1, 15] (should be odd)
    Ps3- Gaussian blur sigma: 1.0 [0.0, 5.0

    Canny edge detection:
    CHs1- lower-threshold: 50 [0, 250]
    CHs2- upper-threshold: 150 [0, 250]  
    
    Hough line detection:
    CHs3- rho-resolution: 1 [1, 5]
    CHs4- theta-resolution (degrees): 1 [0.5, 5]
    CHs5- threshold: 50 [10, 200]
    CHs6- min-line-length: 10 [5, 50]
    CHs7- max-line-gap: 30 [5, 100]
    
    Divider initialization:
    Ds1- n lines: 64 [1, 2048]
    Ds2- band_width_px: 10 [1, 50]  # Extend a band this wide around each line, other line's pixels in band "support" this line.



App layout in 2 tabs:
   1. Line finding.
   2. Divider initialization.
   
Tab 1: Line finding has 3 views
    preprocessing
    canny edges 
    hough lines (combined in one tab)
    

    Canny edges + Hough lines:
    +----------------------------------------------+
    |                                              |
    |                                              |
    |           [image]                            |
    |                                              |
    |                                              |
    +----------------------------------------------+
    | [Ps1--]    [CHs1--]   [CHs4----]  [CHs7----] |
    | [Ps2---]   [CHs2--]   [CHs5----]             |
    | [Ps3---]   [CHs3--]   [CHs6----]      | 
    |{show pre} {show edg}  {show lines}           |                        |
    +----------------------------------------------+

    "Show" buttons switch views in the image area.
    Changes to other params (preprocessing or canny edges) automatically update the image display.


          
    Divider initialization:
    +-----------------------------------+
    |                                   |
    |                                   |
    |           [image]                 |
    |                                   |
    |                                   |
    +-----------------------------------+
    |  [Ds1-----N-lines--------------]  |
    |     [Ds2----]        [save]       |    
    +-----------------------------------+

"""
    
from cProfile import label
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

import cv2
import numpy as np
from PIL import Image, ImageTk
import logging
import os
import json
from tkinter import filedialog
import pickle
from image_net import NNetImage

import sys
from enum import IntEnum
from find_lines import trim_image

class LineMode(IntEnum):
    HOUGH_LINES = 1
    LSD_LINES = 2
    
    
LAYOUT = {'dims': {'min_size_wh': (875, 480),
                   #'tab_image_size_wh': (800, 600),
                   'ctrl_height_px': 275,
                   'status_height_px': 30,
                   'canny_x_split_rel': 0.3,
                   'hough_x_split_rel': 0.7,},
          'fonts': {'tabs': {'name': 'TNotebook.Tab','params': {'family': 'Arial', 'size': 14, 'weight': 'bold'}},
                    'sliders': {'name': 'TScale','params': {'family': 'Arial', 'size': 12}},
                    'buttons': {'name': 'TButton','params': {'family': 'Arial', 'size': 12, 'weight': 'bold'}},
                    'checkButtons': {'name': 'TCheckbutton','params': {'family': 'Arial', 'size': 14},},
                    'status': {'name': 'TLabel','params': {'family': 'Arial', 'size': 12}},
                    'headers': {'name': 'TLabel','params': {'family': 'Arial', 'size': 16, 'weight': 'bold'}},
                    },
          
          
          'tab_grid': dict(ctrl_rows=3, ctrl_cols=3,  # Layout of tab control/image area
                           status_rows=1, status_cols=3,
                           img_rows=10, img_cols=3),
          
          'tabs': {
              'line_finding': {
                  'slider_params': [
                      {'row_col': (1, 0), 'name': 'pre_pyr_down_levels', 'label': 'Num. 2X downsamples', 'type': 'int', 'default': 1, 'range': (0, 10), 'length': 200},
                      {'row_col': (2, 0), 'name': 'pre_gaussian_blur_ksize', 'label': 'Gaussian blur, kernel size', 'type': 'int', 'default': 30, 'range': (1, 50), 'odd_only': True, 'length': 200},
                      {'row_col': (3, 0), 'name': 'pre_gaussian_blur_sigma', 'label': 'Gaussian blur, sigma', 'type': 'float', 'default': 1.0, 'range': (0.0, 6.0), 'length': 200},
                  
                      {'row_col': (1, 1), 'name': 'canny_lower_thresh', 'label': 'lower-threshold', 'type': 'int', 'default': 50, 'range': (0, 1000), 'length': 200},
                      {'row_col': (2, 1), 'name': 'canny_upper_thresh', 'label': 'upper-threshold', 'type': 'int', 'default': 150, 'range': (0, 1000), 'length': 200},
                      
                      {'row_col': (1, 2), 'name': 'hough_rho_res', 'label': 'rho-res', 'type': 'int', 'default': 1, 'range': (1, 15), 'length': 200, 'tag': LineMode.HOUGH_LINES},
                      {'row_col': (2, 2), 'name': 'hough_theta_res_deg', 'label': 'theta-res (deg)', 'type': 'float', 'default': 1.0, 'range': (0.5, 15.0), 'length': 200, 'tag': LineMode.HOUGH_LINES},
                      {'row_col': (3, 2), 'name': 'hough_threshold', 'label': 'threshold', 'type': 'int', 'default': 50,  'range': (1, 2000), 'length': 200, 'tag': LineMode.HOUGH_LINES},
                      {'row_col': (1, 3), 'name': 'hough_min_line_length', 'label': 'min-line-length', 'type': 'int',  'default': 10,  'range': (3, 50), 'length': 200, 'tag': LineMode.HOUGH_LINES},
                      {'row_col': (2, 3), 'name': 'hough_max_line_gap',    'label': 'max-line-gap',    'type': 'int',  'default': 5,  'range': (0, 30), 'length': 200, 'tag': LineMode.HOUGH_LINES},
                    #   {'row_col': (1, 2), 'name': 'lsd_scale', 'label': 'image scale', 'type': 'float', 'default': .8, 'range': (0.0, 1.0), 'length': 200, 'tag': LineMode.LSD_LINES},
                    #   {'row_col': (1, 2), 'name': 'lsd_sigma_scale', 'label': 'sigma scale', 'type': 'float', 'default': 0.6, 'range': (0.0, 10.0), 'length': 200, 'tag': LineMode.LSD_LINES},
                    #   {'row_col': (1, 2), 'name': 'lsd_quant', 'label': 'quantization', 'type': 'float', 'default': 2.0, 'range': (0.0, 10.0), 'length': 200, 'tag': LineMode.LSD_LINES},
                    #   {'row_col': (1, 2), 'name': 'lsd_ang_th', 'label': 'angle thresh.', 'type': 'float', 'default': 22.5, 'range': (0.0, 180.0), 'length': 200, 'tag': LineMode.LSD_LINES},
                    #   {'row_col': (1, 2), 'name': 'lsd_log_eps', 'label': 'log epsilon', 'type': 'float', 'default': 0.0, 'range': (-10.0, 10.0), 'length': 200, 'tag': LineMode.LSD_LINES},
                    #   {'row_col': (1, 2), 'name': 'lsd_n_bins', 'label': 'N bins', 'type': 'int', 'default': 1024, 'range': (10, 2048), 'length': 200, 'tag': LineMode.LSD_LINES},
                      
                  ],
                  'buttons': [{'row_col':(3,3), 'name': 'load_params', 'label': 'load params'},
                              {'row_col':(4,3), 'name': 'save_params', 'label': 'save params'},],
                  'toggles': [{'row_col':(3,1), 'name': 'show_preprocessing', 'label': 'show image'},
                              {'row_col':(4,1), 'name': 'show_canny_edges', 'label': 'show edges'},
                              {'row_col':(4,2), 'name': 'show_lines', 'label': 'show lines'}],                  
                  'headers': [{'row_col': (0,0), 'label': 'Preprocessing','colspan': 1},
                              {'row_col': (0,1), 'label': 'Canny Edges','colspan': 1},
                              {'row_col': (0,2), 'label': 'Hough Line Detection','colspan': 2},]
                  },
                'divider_initialization': {
                    'slider_params': [
                        {'row_col': (1, 0), 'colspan': 3,'name': 'div_n_lines', 'label': 'N Line Dividers', 'type': 'int', 'default': 64, 'range': (1, 512), 'length': 200},
                        {'row_col': (2, 0), 'colspan': 3,'name': 'n_structure_units', 'label': 'N Structure units', 'type': 'int', 'default': 64, 'range': (1, 512), 'length': 200},
                        {'row_col': (3, 0), 'colspan': 3,'name': 'n_color_units', 'label': 'N Color units', 'type': 'int', 'default': 64, 'range': (1, 512), 'length': 200},
                        
                        {'row_col': (1, 1), 'colspan': 3,'name': 'ang_size', 'label': 'Max angle diff', 'type': 'float', 'default': .30, 'range': (.01, 3.0), 'length': 200},
                        {'row_col': (2, 1), 'colspan': 3,'name': 'dist_size', 'label': 'Max distance', 'type': 'float', 'default': .1, 'range': (.001, .03), 'length': 200},
                        {'row_col': (3, 1), 'colspan': 3,'name': 'min_line_len', 'label': 'Min line length', 'type': 'float', 'default': .00, 'range': (0.0, .5), 'length': 200},
                        ],
                  'headers': [{'row_col': (0,0), 'label': 'Network Architecture','colspan': 1},
                              {'row_col': (0,1), 'label': 'Edge selection','colspan': 1},],
                  
                    'buttons': [{'row_col':(1,4), 'name': 'save_divider_init', 'label': 'save'},
                                {'row_col':(2,4), 'name': 'compute_dividers', 'label': 'compute'},],
                    }
          }
}



def pyr_down(image):
    return( (image[::2, ::2].astype(int)+image[1::2, ::2]+image[::2, 1::2]+ image[1::2, 1::2]) //4).astype(np.uint8)



def downsample(image, levels):
    image = trim_image(image, n_pow_2=levels)
    for _ in range(levels):
        image = pyr_down(image)
    return image
def upsample(image, levels):    
    block_template = np.ones((2,2),np.uint8)
    for _ in range(levels):
        image = cv2.merge((np.kron(image[:,:,0], block_template),
                          np.kron(image[:,:,1], block_template),
                          np.kron(image[:,:,2], block_template)))
    return image.astype(np.uint8)
def blur(image, ksize, sigma):
    if sigma==0.0:
        return image
    if ksize % 2 == 0:
        ksize += 1  # make odd
    logging.info(f"Blurring image with ksize={ksize}, sigma={sigma}")
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def get_canny_edges(image, lower_thresh, upper_thresh):
    return cv2.Canny(image, lower_thresh, upper_thresh)

def get_hough_lines(edges, rho_res, theta_res_deg, threshold, min_line_length,   max_line_gap):
    theta_res = np.deg2rad(theta_res_deg)
    logging.info("Finding Hough lines in image with shape %s" % (edges.shape,))
    lines = cv2.HoughLinesP(edges, rho_res, theta_res, threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    return lines



class ViewMode(IntEnum):
    # For tab 1, line finding.
    PREPROCESSING = 1
    CANNY_EDGES = 2

class ExploreLinesApp(tk.Tk):
    def __init__(self, image_path):
        super().__init__()
        self.title("Explore Lines App")
        self.mode = ViewMode.PREPROCESSING
        self.showing_lines = False
        self._cur_tab = 'line_finding'
        self._line_mode = LineMode.LSD_LINES
        
        self._mouse_pos = None  # relative logical pos in line space, and pixel pos in image display
        self._mouse_scale = None
        
        
        self._mouseover_lines = []  
        
        
        
        
        self._lsd = lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)

        self.image = cv2.imread(image_path)
        logging.info(f"Loaded image from {image_path} with shape {self.image.shape}")
        
        width = max(LAYOUT['dims']['min_size_wh'][0], self.image.shape[1])
        height = max(LAYOUT['dims']['min_size_wh'][1], self.image.shape[0] + LAYOUT['dims']['ctrl_height_px'])

        self.geometry(f"{width}x{height}")
        logging.info(f"App window size set to: {width}x{height}")
        
        # Create a style object
        style = ttk.Style()
        style.theme_use('clam')
        
        # # Configure font for buttons:
        button_font = tkFont.Font(**LAYOUT['fonts']['buttons']['params'])
        style.configure(LAYOUT['fonts']['buttons']['name'], font=button_font)
        
        # # Configure font for tabs:
        # tab_font = tkFont.Font(**LAYOUT['fonts']['tabs']['params'])
        # style.configure(LAYOUT['fonts']['tabs']['name'], font=tab_font)
            
        custom_font = tkFont.Font(**LAYOUT['fonts']['tabs']['params'])

        # Configure the 'TNotebook.Tab' style to use the custom font
        style.configure("TNotebook.Tab", font=custom_font)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)
        # Configure callback for tab change:
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
        
        self.tabs = {}
        
        for tab_name, tab_config in LAYOUT['tabs'].items():
            tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(tab_frame, text=tab_name.replace('_', ' ').title())
            tab_config['name'] = tab_name
            self.tabs[tab_name] = self.create_tab_content(tab_frame,  tab_config)
            
        # self.on_toggle('line_finding', 'hough_lines', True)
    
        self.on_toggle('line_finding', 'show_preprocessing', True)
        
        # set slider labels by changing the values (to their defaults)
        for tab_name, tab in self.tabs.items():
            for slider in tab['sliders']:
                slider['scale'].set(getattr(self, slider['name']).get())
        
        self._init_image_pipeline()
        
        self.update_image_display()
        
        # set a callback for window resize events
        # self.bind("<Configure>", self.on_resize)    
        
    # def on_resize(self, event):
    #     self.update_image_display()
        # logging.info("Window resized to: %dx%d" % (event.width, event.height))
        
        
    def on_tab_change(self, event):
        selected_tab = event.widget.tab(event.widget.index("current"))["text"].lower().replace(' ', '_')
        logging.info(f"Switched to tab: {selected_tab}")
        self._cur_tab = selected_tab
        self.update_image_display()
        
    def save_params(self, tab_name):
        """
        Open save-file dialog to save current params to a json file.
        """
        tab = self.tabs[tab_name]
        params = {}
        for slider in tab['sliders']:
            params[slider['name']] = getattr(self, slider['name'])
        
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                                                 title="Save Parameters As")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=4)
            logging.info(f"Saved parameters to {file_path}")
        
    def load_params(self, tab_name):
        """
        Open open-file dialog to load params from a json file.
        """

        file_path = filedialog.askopenfilename(defaultextension=".json",
                                               filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                                               title="Load Parameters From")
        if file_path and os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                params = json.load(f)
            tab = self.tabs[tab_name]
            for slider in tab['sliders']:
                if slider['name'] in params:
                    value = params[slider['name']]
                    slider = [s for s in self.tabs[tab_name]['sliders'] if s['name'] == slider['name']][0]
                    slider['scale'].set(value)  # update the scale position
            logging.info(f"Loaded parameters from {file_path}")
            # Update the image display after loading new params
            self._update_preprocessing()
            self._update_canny_edges()
            self._update_hough_lines()
            self.update_image_display()

    def render_hough_lines(self):
        render_over_orig = self.tabs['line_finding']['toggles'][0]['var'].get()
        valid = self.pipeline['hough_lines_valid']if 'shough_lines_valid' in self.pipeline else [True]* len(self.pipeline['hough_lines'])
        img = self.pipeline['original'].copy() if render_over_orig else self.pipeline['canny_edges_disp'].copy()
        if self.pipeline['hough_lines'] is not None:
            n_downsamples = self.pipeline['n_downsamples']
            for line,valid in zip(self.pipeline['hough_lines'], valid):
                if not valid:
                    continue
                x1, y1, x2, y2 = line[0].astype(float) * (2**n_downsamples)
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, cv2.LINE_AA)
        logging.info("Rendered %i lines on image" % (0 if self.pipeline['hough_lines'] is None else len(self.pipeline['hough_lines'])))
        return img

    def _update_preprocessing(self):
        if not hasattr(self, 'pipeline'):
            return
        n_downsamples = self.tabs['line_finding']['sliders'][0]['var'].get()
        kernel_size = self.tabs['line_finding']['sliders'][1]['var'].get()
        kernel_sigma = self.tabs['line_finding']['sliders'][2]['var'].get()
        logging.info("Updating preprocessed image with n_downsamples=%s, kernel_size=%s, kernel_sigma=%s" % (n_downsamples, kernel_size, kernel_sigma)    )
        self.pipeline['original'] = self.image.copy()
        self.pipeline['n_downsamples'] = n_downsamples
        self.pipeline['downsampled'] = downsample(self.pipeline['original'], n_downsamples)
        # upsample all display images to original size for consistent display
        self.pipeline['downsampled_disp'] = upsample(self.pipeline['downsampled'], n_downsamples)
        self.pipeline['blurred'] = blur(self.pipeline['downsampled'], kernel_size, kernel_sigma)
        self.pipeline['blurred_disp'] = upsample(self.pipeline['blurred'], n_downsamples)
        self._set_status()
        
        
    def _set_status(self):
        status_str = "Image size:  %d x %d (after downsampling:  %d x %d) - Num Hough lines: %d" % (self.pipeline['original'].shape[1], self.pipeline['original'].shape[0],
                                                                           self.pipeline['downsampled'].shape[1], self.pipeline['downsampled'].shape[0],
                                                                           0 if 'hough_lines' not in self.pipeline or self.pipeline['hough_lines'] is None else len(self.pipeline['hough_lines']))    
        self.tabs['line_finding']['status_label'].config(text=status_str)
                
    def _init_image_pipeline(self):     
        self.aspect = self.image.shape[1] / self.image.shape[0]
        if self.aspect > 1.0:
            self.x_scale = 1.0
            self.y_scale = 1.0 / self.aspect
        else:
            self.x_scale = self.aspect
            self.y_scale = 1.0
            
            
        self.pipeline = {}
        # import ipdb; ipdb.set_trace()
        self._update_preprocessing()
        self._update_canny_edges()
        self._update_hough_lines()
        self._update_dividers()
        self._update_divider_images()
        
        
    def _update_canny_edges(self):
        if not hasattr(self, 'pipeline'):
            return
        lower_thresh = self.tabs['line_finding']['sliders'][3]['var'].get()
        upper_thresh = self.tabs['line_finding']['sliders'][4]['var'].get()
        logging.info("Updating Canny edges with lower_thresh=%s, upper_thresh=%s" % (lower_thresh, upper_thresh)    )
        self.pipeline['canny_edges'] = get_canny_edges(self.pipeline['blurred'], lower_thresh, upper_thresh)
        n_downsamples = self.tabs['line_finding']['sliders'][0]['var'].get()
        self.pipeline['canny_edges_disp'] = upsample(cv2.cvtColor(self.pipeline['canny_edges'], cv2.COLOR_GRAY2BGR),
                                                    n_downsamples)
        
        
    def _update_hough_lines(self):
        if not hasattr(self, 'pipeline'):
            return
        hough_rho_res = self.tabs['line_finding']['sliders'][5]['var'].get()
        hough_theta_res_deg = self.tabs['line_finding']['sliders'][6]['var'].get()
        hough_threshold = self.tabs['line_finding']['sliders'][7]['var'].get()
        hough_min_line_length = self.tabs['line_finding']['sliders'][8]['var'].get()
        hough_max_line_gap = self.tabs['line_finding']['sliders'][9]['var'].get()

        self.pipeline['hough_lines'] = get_hough_lines(self.pipeline['canny_edges'],
                                                         hough_rho_res,
                                                        hough_theta_res_deg,
                                                        hough_threshold,
                                                        hough_min_line_length,
                                                        hough_max_line_gap)
        logging.info("Detected %i Hough lines." % (0 if self.pipeline['hough_lines'] is None else len(self.pipeline['hough_lines'])))
        
        self.pipeline['hough_line_params'] = self._calc_line_params()
        
        self.pipeline['hough_lines_disp'] = self.render_hough_lines()
        self._set_status()
    
    def pixel_to_logical(self, size_wh, x, y):
        w, h = size_wh
        lx = (x / w - 0.5) * self.x_scale
        ly = (y / h - 0.5) * self.y_scale
        return lx, ly
    
    def logical_to_pixel(self, size_wh, lx, ly):
        w, h = size_wh
        x = int((lx / self.x_scale + 0.5) * w)
        y = int((ly / self.y_scale + 0.5) * h)
        return x, y
        
    def _calc_line_params(self):
        """
        Convert line coordinates to logical.
        For every line (x0, y0) - (x1, y1) compute (angle, dist) from center of image and (legnth).
        """
        if 'hough_lines' not in self.pipeline:
            return None

        size_wh = self.pipeline['canny_edges'].shape[:2][::-1]
        cx, cy = np.zeros(2)

        line_params = []

        for line in self.pipeline['hough_lines']:
            x0, y0, x1, y1 = line[0]
            x0, y0 = self.pixel_to_logical(size_wh, x0, y0)
            x1, y1 = self.pixel_to_logical(size_wh, x1, y1)
            angle = np.arctan2(y1 - y0, x1 - x0)
            # dist = np.sqrt((cx - (x0 + x1) // 2) ** 2 + (cy - (y0 + y1) // 2) ** 2)
            
            num = (x0*y1 - x1*y0)
            denom = np.sqrt((y1 - y0)**2 + (x1 - x0)**2)
            dist = num / denom  # signed distance from origin to line            
            length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            line_params.append((angle, dist, length))
        line_params = np.array(line_params)
        logging.info(f"Calculated logical parameters for {len(line_params)} lines.")
        return line_params

    def _update_dividers(self):
        
        if not hasattr(self, 'pipeline'):
            return
        # filter lines by length
        min_line_len = self.tabs['divider_initialization']['sliders'][5]['var'].get()
        lengths= self.pipeline['hough_line_params'][:,2]
        
        max_ang = self.tabs['divider_initialization']['sliders'][3]['var'].get()
        max_dist = self.tabs['divider_initialization']['sliders'][4]['var'].get()

        
        valid_lengths=[lengths[l_i] >= min_line_len for l_i, line in enumerate(self.pipeline['hough_lines']) ]
        self.pipeline['hough_lines_valid'] = valid_lengths
        logging.info(f"Filtered Hough lines from {len(lengths)} to {np.sum(valid_lengths)} using min_line_len={min_line_len}")
        image = np.zeros_like(self.pipeline['original'])
        w,h = image.shape[1], image.shape[0]

        # filter moused-over lines, if any
        if self._mouse_pos is not None:
            dists = self.pipeline['hough_line_params'][:,1]
            angles = np.rad2deg(self.pipeline['hough_line_params'][:,0])
            angle_span = np.min(angles), np.max(angles)
            dist_span = np.min(dists), np.max(dists)
            # angle_x = (angles - angle_span[0]) / (angle_span[1] - angle_span[0]) * w
            # dist_y = (dists - dist_span[0]) / (dist_span[1] - dist_span[0]) * h

            # angles_deg = np.rad2deg(angles)
            mouse_angle = self._mouse_pos['angle_deg']
            close_angles = np.abs(angles - mouse_angle) <= max_ang

            ang = angle_span[1] - angle_span[0]
            # print("________________________", mouse_angle)
            box_width_px = (max_ang * 2 / ang) * w  # +/- max_ang in pixels
            

            mouse_dist = self._mouse_pos['dist']
            # mouse_dist = dist_span[0] + mouse_dist_rel * (dist_span[1] - dist_span[0])
            close_dists = np.abs(dists - mouse_dist) <= max_dist
            box_height_px = (max_dist * 2 / (dist_span[1] - dist_span[0])) * h  # +/- max_dist in pixels
            
            inside_mask = close_angles & close_dists
            # import ipdb; ipdb.set_trace()
            self._mouse_pos['inside_mask'] = inside_mask
            self._mouse_pos['box_wh'] = box_width_px, box_height_px
            
            inside_inds = np.where(inside_mask)[0].tolist()
            self._mouseover_lines = inside_inds
            
    def _update_divider_images(self):
        # TODO:  change color to reflect mouseover

        if hasattr(self, 'pipeline'):
            self.pipeline['dividers_img'] = self.render_dividers()
            self.pipeline['lines_img'] = self.render_line_space()
        
    def render_line_space(self): 
        
        image = np.zeros_like(self.pipeline['original'])
        w,h = image.shape[1], image.shape[0]
        lengths= self.pipeline['hough_line_params'][:,2]

        
        
        rad_range = 2, 15
        length_range=np.min(lengths), np.max(lengths)
        
        def length_to_rad(length):
            length_norm = (length - length_range[0]) / (length_range[1] - length_range[0])
            return int(rad_range[0] + length_norm * (rad_range[1] - rad_range[0]))
        
        logging.info("length range:  %.4f to %.4f" % (np.min(lengths), np.max(lengths)))
        
        box_width_px, box_height_px = self._mouse_pos['box_wh'] if self._mouse_pos is not None else (0,0)
        length_mask = self.pipeline['hough_lines_valid']
        inside_mask = self._mouse_pos['inside_mask'] if self._mouse_pos is not None else np.zeros(len(lengths), dtype=bool)
        angles = self.pipeline['hough_line_params'][:,0]
        dists = self.pipeline['hough_line_params'][:,1]
        angle_span = np.min(angles), np.max(angles)
        dist_span = np.min(dists), np.max(dists)
        angle_x = (angles - angle_span[0]) / (angle_span[1] - angle_span[0]) * w
        dist_y = (dists - dist_span[0]) / (dist_span[1] - dist_span[0]) * h
        
        moused = inside_mask & length_mask

        for i, (x, y, length) in enumerate(zip(angle_x, dist_y, lengths)):
            rad_px = int(length_to_rad(length))
            color = (10, 255, 10) if moused[i] else (10, 10, 255) 
            cv2.circle(image, (int(x), int(y)), rad_px, color, -1, cv2.LINE_AA)
            
        if box_width_px >0 and box_height_px>0:
            mx, my = self._mouse_pos['px']
            top_left = (int(mx - box_width_px//2), int(my - box_height_px//2))
            bottom_right = (int(mx + box_width_px//2), int(my + box_height_px//2))
            cv2.rectangle(image, top_left, bottom_right, (255,255,255), 2, cv2.LINE_AA)
            
            
        logging.info(f"Rendered line space with {len(angles)} lines.")
        #return pyr_down(image)
        # factor = 2/3
        # image = cv2.resize(image, (int(w*factor), int(h*factor)), interpolation=cv2.INTER_AREA)
        
        
        return image

    def render_dividers(self):
        if not hasattr(self, 'pipeline') or self.pipeline['hough_lines'] is None:
            img = np.zeros((100, 100, 3), np.uint8)
        else:
            length_mask = self.pipeline['hough_lines_valid']
            
            # Draw only lines that are both long enough and inside mouseover box
            logging.info("Drawing %i lines on image with shape %s" % (self.pipeline['hough_lines'].shape[0], self.pipeline['hough_lines_disp'].shape))
            inside_mask = self._mouse_pos['inside_mask'] if self._mouse_pos is not None else np.zeros(len(length_mask), dtype=bool)
            line_inds = np.where(length_mask & inside_mask)[0].tolist()
            img = self.pipeline['hough_lines_disp'].copy()
            # import ipdb; ipdb.set_trace()
            
            for line_ind in line_inds:
                
                line = self.pipeline['hough_lines'][line_ind]
                n_downsamples = self.pipeline['n_downsamples']
                x1, y1, x2, y2 = line[0].astype(float) * (2**n_downsamples)
                (x1, y1), (x2,y2) = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.line(img, (x1, y1), (x2, y2), (10, 255,10 ), 2, cv2.LINE_AA)
        w,h = img.shape[1], img.shape[0]

        # factor = 2/3
        # img = cv2.resize(img, (int(w*factor), int(h*factor)), interpolation=cv2.INTER_AREA)
        return img


        
    def create_tab_content(self, tab_frame, tab_config):
        """
        Create the content of a tab based on the provided configuration.
        This uses the grid layout to arrange widgets.
        return dict with tab content
        """
        # Image display area
        img_width, img_height = self.image.shape[1], self.image.shape[0]
        tab_content = {}
        
        if tab_config['name']=='line_finding':
            tab_content['img_label'] = tk.Label(tab_frame)
            tab_content['img_label'].grid(row=0,
                                column=0, 
                                columnspan=LAYOUT['tab_grid']['img_cols'],
                                rowspan=LAYOUT['tab_grid']['img_rows'],
                                sticky='nsew')
        elif tab_config['name']=='divider_initialization':
            tab_content['divider_img_label'] = tk.Label(tab_frame)
            tab_content['divider_img_label'].grid(row=0,
                                column=0, 
                                columnspan=2,
                                rowspan=LAYOUT['tab_grid']['img_rows'],
                                sticky='nsew')
            tab_content['lines_img_label'] = tk.Label(tab_frame)
            tab_content['lines_img_label'].grid(row=0,
                                column=3, 
                                columnspan=2,
                                rowspan=LAYOUT['tab_grid']['img_rows'],
                                sticky='nsew')
            # bind mouse events to lines_img_label
            tab_content['lines_img_label'].bind("<Motion>", self.on_lines_img_mouse_move)
        else:
            raise ValueError(f"Unknown tab name: {tab_config['name']}")
        
        # status label under control area
        font_params = LAYOUT['fonts']['status']
        tab_content['status_label'] = ttk.Label(tab_frame, text="Status: Ready", anchor='w',
                                                font = tkFont.Font(**font_params['params']))
        tab_content['status_label'].grid(row=LAYOUT['tab_grid']['img_rows'],
                                        column=0,
                                        rowspan=LAYOUT['tab_grid']['status_rows'],
                                        columnspan=LAYOUT['tab_grid']['status_cols'],
                                        sticky='nsew')
        
        # Control area
        ctrl_frame = ttk.Frame(tab_frame, height=LAYOUT['dims']['ctrl_height_px'])
        ctrl_frame.grid(row=LAYOUT['tab_grid']['img_rows'] + LAYOUT['tab_grid']['status_rows'],
                        column=0,
                        columnspan=LAYOUT['tab_grid']['ctrl_cols'],
                        sticky='nsew')
        
        tab_content['sliders'] = []
        for param in tab_config.get('slider_params', []):
            tab_content['sliders'].append(self.create_slider(ctrl_frame, tab_config['name'], param))
            
        tab_content['buttons'] = []
        for button in tab_config.get('buttons', []):
            tab_content['buttons'].append(self.create_button(ctrl_frame, button))
            
        tab_content['toggles'] = []
        for toggle in tab_config.get('toggles', []):
            tab_content['toggles'].append(self.create_toggle(ctrl_frame, tab_config['name'], toggle))
        
        if 'headers' in tab_config:
            for header in tab_config['headers']:
                font_params = LAYOUT['fonts']['headers']
                label = ttk.Label(ctrl_frame, text=header['label'], font=tkFont.Font(**font_params['params']))
                label.grid(row=header['row_col'][0], column=header['row_col'][1], padx=5, pady=5, columnspan=header.get('colspan', 1))
        tab_content['frame'] = tab_frame
        return tab_content
    
    def on_lines_img_mouse_move(self, event):
        widget = event.widget
        x = event.x
        y = event.y

        w = widget.winfo_width()
        h = widget.winfo_height()
        
        # convert to line-param space coords (angles and distances)
        angles = np.rad2deg(self.pipeline['hough_line_params'][:,0])
        ang_max, ang_min = np.max(angles), np.min(angles)

        distances = self.pipeline['hough_line_params'][:,1]
        dist_max, dist_min = np.max(distances), np.min(distances)
        
        mouse_angle = (x / w) * (ang_max - ang_min) + ang_min  # angle param of mouse
        mouse_dist = (y / h)  * (dist_max - dist_min) + dist_min  # distance param of mouse
        
        logging.info(f"Mouse moved to pixel: ({x}, {y}), angle: ({mouse_angle:.4f}, {mouse_dist:.4f})")
        self._mouse_pos = {'px': (x, y), 
                           'angle_deg': mouse_angle,
                           'dist': mouse_dist,}
        # self._filter_lines()
        self._update_dividers()
        self._update_divider_images()
        self.update_image_display()
        
        

    def create_slider(self, parent,tab_name,  param):
        frame = ttk.Frame(parent)
        colspan = param.get('colspan', 1)
        frame.grid(row=param['row_col'][0], column=param['row_col'][1],columnspan=colspan, sticky='w', padx=5, pady=5   )
        style = ttk.Style()
        custom_font = tkFont.Font(**LAYOUT['fonts']['sliders']['params'])
        style.configure(LAYOUT['fonts']['sliders']['name'], font=custom_font)
        slider = {}
        slider['label'] = ttk.Label(frame, text=param['label'], font=custom_font)
        slider['label'].pack(side='top', anchor='w')
        if param['type'] == 'int':
            var_type = tk.IntVar
            slider['format_str'] = f"{param['label']}=%d"
            slider['val_type'] = int
        else:
            var_type = tk.DoubleVar
            slider['format_str'] = f"{param['label']}=%.2f"
            slider['val_type'] = float
            
        logging.info("Making slider for tab %s named %s with value type %s, default %s" % (self._cur_tab, param['name'], var_type, param['default'])    )
        var = var_type(value=param['default'])
        slider['scale'] = ttk.Scale(frame, from_=param['range'][0], length=param['length'], command=lambda val, name=param['name']: self.slider_changed(tab_name, name, val),
                           to=param['range'][1], orient='horizontal', variable=var)
        slider['scale'].pack(side='top', anchor='w')
        
        
        
        setattr(self, param['name'], var)
        slider['name'] = param['name']
        slider['var'] = var
        return slider
    
    
    def slider_changed(self,tab_name, slider_name, value):
        logging.info(f"Tab {tab_name} Slider {slider_name} changed to {value}, type {type(value)}")
        setattr(self, slider_name, value)
        # Update slider label to show value:
        
        slider = [s for s in self.tabs[tab_name]['sliders'] if s['name'] == slider_name][0]
        value = slider['val_type'](float(value))
        slider_str = slider['format_str'] % (value,)
        slider['label'].config(text=slider_str)
        
        if slider_name in ['pre_pyr_down_levels', 'pre_gaussian_blur_ksize', 'pre_gaussian_blur_sigma']:
            self._update_preprocessing()
            self._update_canny_edges()
            self._update_hough_lines()
        elif slider_name in ['canny_lower_thresh', 'canny_upper_thresh']:
            self._update_canny_edges()
            self._update_hough_lines()
        elif slider_name in ['hough_rho_res', 'hough_theta_res_deg', 'hough_threshold',
                             'hough_min_line_length', 'hough_max_line_gap']:
            self._update_hough_lines()
            self._update_dividers()
            self._update_divider_images()
        
        elif slider_name in ['div_n_lines']:
            #self._update_dividers()
            logging.info("Divider slider changed, but not updating dividers yet (click apply).")
            
        elif slider_name in ['min_line_len', 'ang_size', 'dist_size']:
            logging.info("Line mode specific slider changed.")
            # self._filter_lines()
            self._update_dividers()
            self._update_divider_images()
            
        
        
        
        self.update_image_display()
        
        
    def create_toggle(self, parent, tab_name, toggle_info):
        toggle_button = {}
        var = tk.BooleanVar()
        style = ttk.Style()
        custom_font = tkFont.Font(**LAYOUT['fonts']['checkButtons']['params'])
        style.configure(LAYOUT['fonts']['checkButtons']['name'], font=custom_font)

        toggle_button['button'] = ttk.Checkbutton(parent, text=toggle_info['label'], variable=var,
                                       command=lambda name=toggle_info['name'], v=var: self.on_toggle(tab_name, name, v.get()))
        toggle_button['button'].grid(row=toggle_info['row_col'][0], sticky='w',
                           column=toggle_info['row_col'][1], padx=5, pady=5)
        toggle_button['name'] = toggle_info['name']
        toggle_button['var'] = var
        return toggle_button
        
    def on_toggle(self,tab_name, toggle_name, state):
        logging.info(f"Toggle {toggle_name} set to {state}")
                        
        toggle = [t for t in self.tabs[tab_name]['toggles'] if t['name'] == toggle_name]
        if len(toggle) == 0:
            raise ValueError(f"Unknown toggle name: {toggle_name}")
        toggle = toggle[0]
        toggle['var'].set(state)    
    
        if toggle_name == 'show_preprocessing' and state:
            self.mode = ViewMode.PREPROCESSING
            toggle['button'].state(['selected'])
            edge_toggle = [t for t in self.tabs[tab_name]['toggles'] if t['name'] == 'show_canny_edges'][0]
            edge_toggle['button'].state(['!selected']) 
            edge_toggle['var'].set(False)
        elif toggle_name == 'show_canny_edges' and state:
            self.mode = ViewMode.CANNY_EDGES
            toggle['button'].state(['selected'])
            pre_toggle = [t for t in self.tabs[tab_name]['toggles'] if t['name'] == 'show_preprocessing'][0]
            pre_toggle['button'].state(['!selected'])
            pre_toggle['var'].set(False)
            
        elif toggle_name == 'show_lines':
            self.showing_lines = state
            logging.info(f"Setting showing_lines to {self.showing_lines}")
            
            if state:
                toggle['button'].state(['selected'])
            else:
                toggle['button'].state(['!selected'])
                
        elif toggle_name in ["hough_lines", "lsd_lines"]:

            self.set_line_mode(toggle_name, state)
                
        if self.showing_lines:
            self.pipeline['hough_lines_disp'] = self.render_hough_lines()
                
            
        self.update_image_display()
        
    def set_line_mode(self, line_mode, state):
        logging.info(f"Setting line mode to {line_mode} with state {state}")
        
        
        
        if line_mode == 'hough_lines' and state:
            hl_toggle = [t for t in self.tabs['line_finding']['toggles'] if t['name'] == 'hough_lines'][0]
            hl_toggle['button'].state(['selected'])
            hl_toggle['var'].set(True)
            lsd_toggle = [t for t in self.tabs['line_finding']['toggles'] if t['name'] == 'lsd_lines'][0]
            lsd_toggle['button'].state(['!selected'])
            lsd_toggle['var'].set(False)
            logging.info("Hough lines mode selected.")
            
            # Deactivate LSD-specific sliders
            for slider in self.tabs['line_finding']['sliders']:
                if 'tag' in slider and slider['tag'] == LineMode.LSD_LINES:
                    slider['scale'].state(['disabled'])
                    
            # activate Hough-specific sliders
            for slider in self.tabs['line_finding']['sliders']:
                if 'tag' in slider and slider['tag'] == LineMode.HOUGH_LINES:
                    slider['scale'].state(['!disabled'])
            
            
            
        elif line_mode == 'lsd_lines' and state:
            lsd_toggle = [t for t in self.tabs['line_finding']['toggles'] if t['name'] == 'lsd_lines'][0]
            lsd_toggle['button'].state(['selected'])
            lsd_toggle['var'].set(True)
            hl_toggle = [t for t in self.tabs['line_finding']['toggles'] if t['name'] == 'hough_lines'][0]
            hl_toggle['button'].state(['!selected'])
            hl_toggle['var'].set(False)
            logging.info("LSD lines mode selected.")
            
            # activate LSD-specific sliders
            for slider in self.tabs['line_finding']['sliders']:
                if 'tag' in slider and slider['tag'] == LineMode.LSD_LINES:
                    slider['scale'].state(['!disabled'])
                    
            # Deactivate Hough-specific sliders
            for slider in self.tabs['line_finding']['sliders']:
                if 'tag' in slider and slider['tag'] == LineMode.HOUGH_LINES:
                    slider['scale'].state(['disabled'])
                    
                    
        else:
            logging.info("No line mode selected.")
        
    def create_button(self, parent, button_info):
        button={'button': ttk.Button(parent, text=button_info['label'], command=lambda: self.on_button_click(button_info['name']),
                                     padding=(5, 1))}
        button['button'].grid(row=button_info['row_col'][0], 
                 column=button_info['row_col'][1], padx=5, pady=5)
        button['name'] = button_info['name']
        
        return button
        
    def on_button_click(self, button_name):
        if button_name == 'save_divider_init':
            self.save_divider_initialization()
        elif button_name == 'save_params':
            self.save_params(self._cur_tab)
        elif button_name == 'load_params':
            self.load_params(self._cur_tab)
            self._init_image_pipeline()
        elif button_name =='compute_dividers':
            logging.info("Computing dividers...")
            self._update_dividers()
            self._update_divider_images()
            self.update_image_display()
            

        else:
            raise ValueError(f"Unknown button action: {button_name}")
            
    def save_divider_initialization(self):
        
        n_lines = self.tabs['divider_initialization']['sliders'][0]['var'].get()
        n_struct = self.tabs['divider_initialization']['sliders'][1]['var'].get()
        n_color = self.tabs['divider_initialization']['sliders'][2]['var'].get()
        logging.info(f"Saving diviers with n_lines={n_lines}, n_struct={n_struct}, n_color={n_color}")
        
        n_div = {'linear': n_lines,
                 'circular': 0,
                 'normal': 0}
        
        image = self.pipeline['original']
        
        net_image = NNetImage(image=image[:,:,::-1], n_div=n_div, n_structure=n_struct, n_hidden=n_color)
        save_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                                 filetypes=[("NNetImage (pickle) files", "*.pkl"), ("All files", "*.*")],
                                                 title="Save Divider Initialization As")
        if save_path:
            net_image.save_state(save_path)
            logging.info(f"Saved divider initialization to {save_path}")

        self.update_image_display()
        
    def update_image_display(self):
        logging.info("Updating image display for tab %s" % (self._cur_tab,)    )
        if not hasattr(self, 'pipeline'):
            logging.warning("Pipeline not initialized yet, skipping image update")
            return
        
        if self._cur_tab == 'line_finding':
            if self.showing_lines:
                image = self.pipeline['hough_lines_disp']
            else:
                if self.mode == ViewMode.PREPROCESSING:
                    image = self.pipeline['blurred_disp']
                elif self.mode == ViewMode.CANNY_EDGES:
                    image = self.pipeline['canny_edges_disp']
                else:
                    raise ValueError(f"Unknown view mode: {self.mode}")
        
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

            self.tabs[self._cur_tab]['img_label'].config(image=img_tk)
            self.tabs[self._cur_tab]['img_label'].image = img_tk
                
                
        elif self._cur_tab == 'divider_initialization':
            
            div_image = self.pipeline['dividers_img']
            line_image = self.pipeline['lines_img']
            div_img_rgb = cv2.cvtColor(div_image, cv2.COLOR_BGR2RGB)
            line_img_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

            div_img_tk = ImageTk.PhotoImage(Image.fromarray(div_img_rgb))
            line_img_tk = ImageTk.PhotoImage(Image.fromarray(line_img_rgb))

            self.tabs[self._cur_tab]['divider_img_label'].config(image=div_img_tk)
            self.tabs[self._cur_tab]['divider_img_label'].image = div_img_tk
            self.tabs[self._cur_tab]['lines_img_label'].config(image=line_img_tk)
            self.tabs[self._cur_tab]['lines_img_label'].image = line_img_tk
            
        
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python explore_lines.py <image_path>")
        sys.exit(1)
    
    app = ExploreLinesApp( sys.argv[1])
    app.mainloop()