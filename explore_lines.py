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
LAYOUT = {'dims': {'min_size_wh': (875, 480),
                   #'tab_image_size_wh': (800, 600),
                   'ctrl_height_px': 275,
                   'status_height_px': 30,
                   'canny_x_split_rel': 0.3,
                   'hough_x_split_rel': 0.7,},
          'fonts': {'tabs': {'name': 'TNotebook.Tab','params': {'family': 'Arial', 'size': 14, 'weight': 'bold'}},
                    'sliders': {'name': 'TScale','params': {'family': 'Arial', 'size': 12}},
                    'buttons': {'name': 'TButton','params': {'family': 'Arial', 'size': 14, 'weight': 'bold'}},
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
                      
                      {'row_col': (1, 2), 'name': 'hough_rho_res', 'label': 'rho-resolution', 'type': 'int', 'default': 1, 'range': (1, 15), 'length': 200},
                      {'row_col': (2, 2), 'name': 'hough_theta_res_deg', 'label': 'theta-resolution (degrees)', 'type': 'float', 'default': 1.0, 'range': (0.5, 15.0), 'length': 200},
                      {'row_col': (3, 2), 'name': 'hough_threshold', 'label': 'threshold', 'type': 'int', 'default': 50,  'range': (1, 2000), 'length': 200},
                      {'row_col': (1, 3), 'name': 'hough_min_line_length', 'label': 'min-line-length', 'type': 'int',  'default': 25,  'range': (5, 50), 'length': 200},
                      {'row_col': (2, 3), 'name': 'hough_max_line_gap',    'label': 'max-line-gap',    'type': 'int',  'default': 5,  'range': (0, 30), 'length': 200},
                  ],
                  
                  'toggles': [{'row_col':(3,1), 'name': 'show_preprocessing', 'label': 'show image'},
                              {'row_col':(4,1), 'name': 'show_canny_edges', 'label': 'show edges'},
                              {'row_col':(4,2), 'name': 'show_hough_lines', 'label': 'show lines'},],
                  
                  'headers': [{'row_col': (0,0), 'label': 'Preprocessing','colspan': 1},
                              {'row_col': (0,1), 'label': 'Canny Edges','colspan': 1},
                              {'row_col': (0,2), 'label': 'Hough Line Detection','colspan': 2},]
                  },
                'divider_initialization': {
                    'slider_params': [
                        {'row_col': (0, 0), 'colspan': 2,'name': 'div_n_lines', 'label': 'Ds1- n lines', 'type': 'int', 'default': 64, 'range': (1, 2048), 'length': 700},
                        {'row_col': (1, 0), 'name': 'div_band_width_px', 'label': 'Ds2- band_width_px', 'type': 'int', 'default': 10, 'range': (1, 50), 'length': 200  },],
                    'buttons': [{'row_col':(1,1), 'name': 'save_divider_init', 'label': 'save'}]       ,
                    'toggles': [{'row_col':(2,0), 'name': 'show_bands', 'label': 'show bands'}],
                    }
          }
}
from cProfile import label
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

import cv2
import numpy as np
from PIL import Image, ImageTk
import logging
import sys
from enum import IntEnum
from find_lines import trim_image




def pyr_down(image):
    return( (image[::2, ::2].astype(int)+image[1::2, ::2]+image[::2, 1::2]+ image[1::2, 1::2]) //4).astype(np.uint8)



def downsample(image, levels):
    image = trim_image(image, n_pow_2=levels)
    for _ in range(levels):
        image = pyr_down(image)
    return image
def upsample(image, levels):    
    block_template = np.ones((2,2),np.uint8)
    print(image.shape, image.dtype)
    for _ in range(levels):
        image = cv2.merge((np.kron(image[:,:,0], block_template),
                          np.kron(image[:,:,1], block_template),
                          np.kron(image[:,:,2], block_template)))
    print(image.shape, image.dtype)
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

def get_hough_lines(edges, rho_res, theta_res_deg, threshold, min_line_length, max_line_gap):
    theta_res = np.deg2rad(theta_res_deg)
    lines = cv2.HoughLinesP(edges, rho_res, theta_res, threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    return lines
def unify_lines(pipeline, n_lines, band_width_px):
    return None

def render_dividers(pipeline, show_bands=False):
    img = np.zeros_like(pipeline['downsampled'])
    text = "Rendered dividers, bands " + ("shown" if show_bands else "hidden")
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img



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
        
        self.tabs = {}
        
        for tab_name, tab_config in LAYOUT['tabs'].items():
            tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(tab_frame, text=tab_name.replace('_', ' ').title())
            tab_config['name'] = tab_name
            self.tabs[tab_name] = self.create_tab_content(tab_frame,  tab_config)
    
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
        

    def render_hough_lines(self):
        render_over_orig = self.tabs['line_finding']['toggles'][0]['var'].get()
        print("RENDERING OVER ORIGINAL IMAGE:   ", render_over_orig)
        
        img = self.pipeline['original'].copy() if render_over_orig else self.pipeline['canny_edges_disp'].copy()
        if self.pipeline['hough_lines'] is not None:
            n_downsamples = self.pipeline['n_downsamples']
            for line in self.pipeline['hough_lines']:
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
        
        status_str = "Image size:  %d x %d (after downsampling:  %d x %d) - Num Hough lines: %d" % (self.pipeline['original'].shape[1], self.pipeline['original'].shape[0],
                                                                           self.pipeline['downsampled'].shape[1], self.pipeline['downsampled'].shape[0],
                                                                           0 if 'hough_lines' not in self.pipeline or self.pipeline['hough_lines'] is None else len(self.pipeline['hough_lines']))    
        self.tabs['line_finding']['status_label'].config(text=status_str)
                
    def _init_image_pipeline(self):     
        self.pipeline = {}
        self._update_preprocessing()
        self._update_canny_edges()
        self._update_hough_lines()
        self._update_dividers()
        
        
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
        
        self.pipeline['hough_lines_disp'] = self.render_hough_lines()
        
    def _update_dividers(self):
        if not hasattr(self, 'pipeline'):
            return
        n_lines = self.tabs['divider_initialization']['sliders'][0]['var'].get()
        band_width_px = self.tabs['divider_initialization']['sliders'][1]['var'].get()
        self.pipeline['divider_lines'] = unify_lines(self.pipeline, n_lines, band_width_px)
        self.pipeline['dividers_img'] = render_dividers(self.pipeline, show_bands=False)
        self.pipeline['dividers_img_with_bands'] = render_dividers(self.pipeline, show_bands=True)
        
    def create_tab_content(self, tab_frame, tab_config):
        """
        Create the content of a tab based on the provided configuration.
        This uses the grid layout to arrange widgets.
        return dict with tab content
        """
        # Image display area
        img_width, img_height = self.image.shape[1], self.image.shape[0]
        tab_content = {}
        tab_content['img_label'] = tk.Label(tab_frame)
        tab_content['img_label'].grid(row=0,
                            column=0, 
                            columnspan=LAYOUT['tab_grid']['img_cols'],
                            rowspan=LAYOUT['tab_grid']['img_rows'],
                            sticky='nsew')
        
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
        
        elif slider_name in ['div_n_lines', 'div_band_width_px']:
            self._update_dividers()
        
        
        
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
            
        elif toggle_name == 'show_hough_lines':
            self.showing_lines = state
            logging.info(f"Setting showing_lines to {self.showing_lines}")
            
            if state:
                toggle['button'].state(['selected'])
            else:
                toggle['button'].state(['!selected'])
                
        if self.showing_lines:
            self.pipeline['hough_lines_disp'] = self.render_hough_lines()
                
            
        self.update_image_display()
        
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

        else:
            raise ValueError(f"Unknown button action: {button_name}")
            
    def save_divider_initialization(self):
        print("Saving divider initialization...")
        # Implement saving logic here
        
        
    def update_image_display(self):
        logging.info("Updating image display")
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
        elif self._cur_tab == 'divider_initialization':
            show_bands_toggle = [t for t in self.tabs['divider_initialization']['toggles'] if t['name'] == 'show_bands'][0]
            show_bands = show_bands_toggle['var'].get()
            if show_bands:
                image = self.pipeline['dividers_img_with_bands']
            else:
                image = self.pipeline['dividers_img']
        
        
        
        
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        # img_pil = img_pil.resize(LAYOUT['dims']['tab_image_size_wh'])
        img_tk = ImageTk.PhotoImage(img_pil)

        self.tabs[self._cur_tab]['img_label'].config(image=img_tk)
        self.tabs[self._cur_tab]['img_label'].image = img_tk
        
        for tab in self.tabs.values():
            tab['img_label'].config(image=img_tk)
            tab['img_label'].image = img_tk
        
        
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python explore_lines.py <image_path>")
        sys.exit(1)
    
    app = ExploreLinesApp( sys.argv[1])
    app.mainloop()