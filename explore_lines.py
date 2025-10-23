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
    | [Ps3---]   [CHs3--]   [CHs6----]   [apply]   | 
    |{show pre} {show edg}  {show lines}           |                        |
    +----------------------------------------------+

    "Show" buttons switch views in the image area.
    "Apply" button runs the hough transform w/current params.
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
LAYOUT = {'dims': {'min_size_wh': (640, 480),
                   #'tab_image_size_wh': (800, 600),
                   'ctrl_height_px': 265,
                   'canny_x_split_rel': 0.3,
                   'hough_x_split_rel': 0.7,},
          'fonts': {'tabs': {'name': 'TNotebook.Tab','params': {'family': 'Arial', 'size': 24, 'weight': 'bold'}},
                    'sliders': {'name': 'TScale','params': {'family': 'Arial', 'size': 12}},
                    'buttons': {'name': 'TButton','params': {'family': 'Arial', 'size': 14, 'weight': 'bold'}},
                    'checkButtons': {'name': 'TCheckbutton','params': {'family': 'Arial', 'size': 14}}
                    },
          
          
          'tab_grid': dict(ctrl_rows=3, ctrl_cols=3,  # Layout of tab control/image area
                           img_rows=10, img_cols=3),
          
          'tabs': {
              'line_finding': {
                  'slider_params': [
                      {'row_col': (1, 0), 'name': 'pre_pyr_down_levels', 'label': 'Num. 2X downsamples', 'type': 'int', 'default': 1, 'range': (0, 10), 'length': 200},
                      {'row_col': (2, 0), 'name': 'pre_gaussian_blur_ksize', 'label': 'Gaussian blur, kernel size', 'type': 'int', 'default': 5, 'range': (1, 15), 'odd_only': True, 'length': 200},
                      {'row_col': (3, 0), 'name': 'pre_gaussian_blur_sigma', 'label': 'Gaussian blur, sigma', 'type': 'float', 'default': 1.0, 'range': (0.0, 5.0), 'length': 200},
                  
                      {'row_col': (1, 1), 'name': 'canny_lower_thresh', 'label': 'Canny lower-threshold', 'type': 'int', 'default': 50, 'range': (0, 250), 'length': 200},
                      {'row_col': (2, 1), 'name': 'canny_upper_thresh', 'label': 'Canny upper-threshold', 'type': 'int', 'default': 150, 'range': (0, 250), 'length': 200},
                      
                      {'row_col': (1, 2), 'name': 'hough_rho_res', 'label': 'Hough rho-resolution', 'type': 'int', 'default': 1, 'range': (1, 5), 'length': 200},
                      {'row_col': (2, 2), 'name': 'hough_theta_res_deg', 'label': 'theta-resolution (degrees)', 'type': 'float', 'default': 1.0, 'range': (0.5, 5.0), 'length': 200},
                      {'row_col': (3, 2), 'name': 'hough_threshold', 'label': 'threshold', 'type': 'int', 'default': 50,  'range': (10, 200), 'length': 200},
                      {'row_col': (1, 3), 'name': 'hough_min_line_length', 'label': 'min-line-length', 'type': 'int',  'default': 10,  'range': (5, 50), 'length': 200},
                      {'row_col': (2, 3), 'name': 'hough_max_line_gap',    'label': 'max-line-gap',    'type': 'int',  'default': 30,  'range': (5, 100), 'length': 200},
                  ],
                  'buttons': [{'row_col':(3,3), 'name': 'apply', 'label': 'apply'},],
                  
                  'toggles': [{'row_col':(4,0), 'name': 'show_preprocessing', 'label': 'show image'},
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

class ViewMode(IntEnum):
    # For tab 1, line finding.
    PREPROCESSING = 1
    CANNY_EDGES = 2
    HOUGH_LINES = 3

class ExploreLinesApp(tk.Tk):
    def __init__(self, image_path):
        super().__init__()
        self.title("Explore Lines App")
        self.mode = ViewMode.PREPROCESSING
        self._cur_tab = 'line_finding'
        
        self.image = cv2.imread(image_path)
        logging.info(f"Loaded image from {image_path} with shape {self.image.shape}")
        self.original_image = self.image.copy()
        
        width = max(LAYOUT['dims']['min_size_wh'][0], self.image.shape[1])
        height = max(LAYOUT['dims']['min_size_wh'][1], self.image.shape[0]) + LAYOUT['dims']['ctrl_height_px']

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
            
        custom_font = tkFont.Font(family="Arial", size=34, weight="bold")

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
        # set mode to preprocessing initially
        
        
        
        
        
        self.update_image_display()
        self.on_toggle('line_finding', 'show_preprocessing', True)
            
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
        
        # Control area
        ctrl_frame = ttk.Frame(tab_frame, height=LAYOUT['dims']['ctrl_height_px'])
        ctrl_frame.grid(row=LAYOUT['tab_grid']['img_rows'],
                        column=0,
                        columnspan=LAYOUT['tab_grid']['ctrl_cols'],
                        sticky='nsew')
        
        tab_content['sliders'] = []
        for param in tab_config.get('slider_params', []):
            tab_content['sliders'].append(self.create_slider(ctrl_frame, param))
            
        tab_content['buttons'] = []
        for button in tab_config.get('buttons', []):
            tab_content['buttons'].append(self.create_button(ctrl_frame, button))
            
        tab_content['toggles'] = []
        for toggle in tab_config.get('toggles', []):
            tab_content['toggles'].append(self.create_toggle(ctrl_frame, tab_config['name'], toggle))
        
        if 'headers' in tab_config:
            for header in tab_config['headers']:
                label = ttk.Label(ctrl_frame, text=header['label'], font=tkFont.Font(size=14, weight='bold'))
                label.grid(row=header['row_col'][0], column=header['row_col'][1], padx=5, pady=5, columnspan=header.get('colspan', 1))
        tab_content['frame'] = tab_frame
        return tab_content

    def create_slider(self, parent, param):
        frame = ttk.Frame(parent)
        colspan = param.get('colspan', 1)
        frame.grid(row=param['row_col'][0], column=param['row_col'][1],columnspan=colspan, sticky='w', padx=5, pady=5   )
        style = ttk.Style()
        custom_font = tkFont.Font(**LAYOUT['fonts']['sliders']['params'])
        style.configure(LAYOUT['fonts']['sliders']['name'], font=custom_font)
        slider = {}
        slider['label'] = ttk.Label(frame, text=param['label'], font=custom_font)
        slider['label'].pack(side='top', anchor='w')
        
        var_type = tk.IntVar if param['type'] == 'int' else tk.DoubleVar
        var = var_type(value=param['default'])
        slider['scale'] = ttk.Scale(frame, from_=param['range'][0], length=param['length'], command=lambda val, name=param['name']: self.slider_changed(name, val),
                           to=param['range'][1], orient='horizontal', variable=var)
        slider['scale'].pack()
        
        
        setattr(self, param['name'], var)
        slider['name'] = param['name']
        slider['var'] = var
        return slider
    
    
    def slider_changed(self, slider_name, value):
        logging.info(f"Slider {slider_name} changed to {value}")
        setattr(self, slider_name, value)
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
        # import ipdb; ipdb.set_trace()
        
        # deactivate other toggles in the same tab
        for toggle in self.tabs[tab_name]['toggles']:
            print( toggle['name'], toggle_name)
            if toggle['name'] != toggle_name:
                toggle['button'].state(['!selected'])
                
                
        toggle = [t for t in self.tabs[tab_name]['toggles'] if t['name'] == toggle_name]
        if len(toggle) == 0:
            raise ValueError(f"Unknown toggle name: {toggle_name}")
        toggle = toggle[0]
    
        if toggle_name == 'show_preprocessing' and state:
            self.mode = ViewMode.PREPROCESSING
            
            toggle['button'].state(['selected'])
        elif toggle_name == 'show_canny_edges' and state:
            self.mode = ViewMode.CANNY_EDGES
            toggle['button'].state(['selected'])
        elif toggle_name == 'show_hough_lines' and state:
            self.mode = ViewMode.HOUGH_LINES
            toggle['button'].state(['selected'])

            
        # elif toggle_name == 'show_canny_edges' and state:
        #     self.mode = ViewMode.CANNY_EDGES
        # elif toggle_name == 'show_hough_lines' and state:
        #     self.mode = ViewMode.HOUGH_LINES
        # else:
        #     raise ValueError(f"Unknown toggle action: {toggle_name}")
        # self.update_image_display()
        
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
        elif button_name == 'apply':
            self.apply_hough_transform()
        elif button_name == 'show_preprocessing':
            self.mode = ViewMode.PREPROCESSING
            self.update_image_display()
        elif button_name == 'show_canny_edges':
            self.mode = ViewMode.CANNY_EDGES
            self.update_image_display()
        elif button_name == 'show_hough_lines':
            self.mode = ViewMode.HOUGH_LINES
            self.update_image_display()
            
        else:
            raise ValueError(f"Unknown button action: {button_name}")
            
    def save_divider_initialization(self):
        print("Saving divider initialization...")
        # Implement saving logic here
        
    def apply_hough_transform(self):
        print("Applying Hough Transform...")
        # Implement Hough transform logic here
        
    def update_image_display(self):
        logging.info("Updating image display")
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
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