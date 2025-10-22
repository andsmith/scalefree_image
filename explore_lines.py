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



App layout in 3 tabs:
    preprocessing
    canny edges + hough lines (combined in one tab)
    and dividers initializations.
    
    Preprocessing: 
    +---------------------------------+
    |                                 |
    |                                 |
    |           [image]               |
    |                                 |
    |                                 |
    |                                 |
    +---------------------------------+
    |  [Ps1--]  [Ps2----]  [Ps3----]  |
    +---------------------------------+


    Canny edges + Hough lines:
    +-----------------------------------+
    |                                   |
    |                                   |
    |           [image]                 |
    |                                   |
    |                                   |
    +----------+------------------------+
    |  Canny   |  Hough      [CHs6----] |
    | [CHs2--] | [CHs4----]  [CHs7----] |
    | [CHs1--] | [CHs3----]  [CHs5----] | 
    +----------+------------------------+
          ^ - hysteresis thresholds move each other so as not to cross.
          
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
                   'ctrl_height_px': 200,
                   'canny_x_split_rel': 0.3,
                   'hough_x_split_rel': 0.7,},
          'fonts': {'tabs': {'name': 'TNotebook.Tab','params': {'family': 'Arial', 'size': 24, 'weight': 'bold'}},
                    'sliders': {'name': 'TScale','params': {'family': 'Arial', 'size': 12}},
                    'buttons': {'name': 'TButton','params': {'family': 'Arial', 'size': 12, 'weight': 'bold'}}},
          
          
          'tab_grid': dict(ctrl_rows=3, ctrl_cols=3,  # Layout of tab control/image area
                           img_rows=10, img_cols=3),
          
          'tabs': {
              'preprocessing': {
                  'slider_params': [
                      {'row_col': (0, 0), 'name': 'pyr_down_levels', 'label': 'Ps1- pyramid down levels', 'type': 'int', 'default': 1, 'range': (0, 10), 'length': 200},
                      {'row_col': (0, 1), 'name': 'gaussian_blur_ksize', 'label': 'Ps2- Gaussian blur kernel size', 'type': 'int', 'default': 5, 'range': (1, 15), 'odd_only': True, 'length': 200},
                      {'row_col': (0, 2), 'name': 'gaussian_blur_sigma', 'label': 'Ps3- Gaussian blur sigma', 'type': 'float', 'default': 1.0, 'range': (0.0, 5.0), 'length': 200},
                  ]
              },
              'canny_hough': {
                  'slider_params': [
                      {'row_col': (0, 0), 'name': 'canny_lower_thresh', 'label': 'CHs1- lower-threshold', 'type': 'int', 'default': 50, 'range': (0, 250), 'length': 200},
                      {'row_col': (1, 0), 'name': 'canny_upper_thresh', 'label': 'CHs2- upper-threshold', 'type': 'int', 'default': 150, 'range': (0, 250), 'length': 200},
                      
                      {'row_col': (0, 2), 'name': 'hough_rho_res', 'label': 'CHs3- rho-resolution', 'type': 'int', 'default': 1, 'range': (1, 5), 'length': 200},
                      {'row_col': (1, 2), 'name': 'hough_theta_res_deg', 'label': 'CHs4- theta-resolution (degrees)', 'type': 'float', 'default': 1.0, 'range': (0.5, 5.0), 'length': 200},
                      {'row_col': (2, 2), 'name': 'hough_threshold', 'label': 'CHs5- threshold', 'type': 'int', 'default': 50,  'range': (10, 200), 'length': 200},
                      {'row_col': (0, 3), 'name': 'hough_min_line_length', 'label': 'CHs6- min-line-length', 'type': 'int',  'default': 10,  'range': (5, 50), 'length': 200},
                      {'row_col': (1, 3), 'name': 'hough_max_line_gap',    'label': 'CHs7- max-line-gap',    'type': 'int',  'default': 30,  'range': (5, 100), 'length': 200},
                  ],
                  'buttons': [{'row_col':(2,3), 'name': 'apply', 'label': 'apply'}]
                },
                'divider_initialization': {
                    'slider_params': [
                        {'row_col': (0, 0), 'colspan': 2,'name': 'n_lines', 'label': 'Ds1- n lines', 'type': 'int', 'default': 64, 'range': (1, 2048), 'length': 700},
                        {'row_col': (1, 0), 'name': 'band_width_px', 'label': 'Ds2- band_width_px', 'type': 'int', 'default': 10, 'range': (1, 50), 'length': 200  },],
                    'buttons': [{'row_col':(1,1), 'name': 'save_divider_init', 'label': 'save'}]       
                    }
          }
}
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

import cv2
import numpy as np
from PIL import Image, ImageTk
import logging
import sys

class ExploreLinesApp(tk.Tk):
    def __init__(self, image_path):
        super().__init__()
        self.title("Explore Lines App")
        
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
            self.tabs[tab_name] = {'frame': tab_frame,  
                                   'img_label': self.create_tab_content(tab_frame, tab_config)}
        
        self.update_image_display()
        
            
    def create_tab_content(self, tab_frame, tab_config):
        """
        Create the content of a tab based on the provided configuration.
        This uses the grid layout to arrange widgets.
        """
        # Image display area
        img_width, img_height = self.image.shape[1], self.image.shape[0]
        self.img_label = tk.Label(tab_frame)
        self.img_label.grid(row=0,
                            column=0, 
                            columnspan=LAYOUT['tab_grid']['img_cols'],
                            rowspan=LAYOUT['tab_grid']['img_rows'],
                            sticky='nsew')
        
        # Control area
        ctrl_frame = ttk.Frame(tab_frame, height=LAYOUT['dims']['ctrl_height_px'])
        ctrl_frame.grid(row=LAYOUT['tab_grid']['img_rows'],
                        column=0,
                        columnspan=LAYOUT['tab_grid']['ctrl_cols'],
                        sticky='ew')
        
        for param in tab_config.get('slider_params', []):
            self.create_slider(ctrl_frame, param)
        
        for button in tab_config.get('buttons', []):
            self.create_button(ctrl_frame, button)
        return self.img_label

    def create_slider(self, parent, param):
        frame = ttk.Frame(parent)
        colspan = param.get('colspan', 1)
        frame.grid(row=param['row_col'][0], column=param['row_col'][1],columnspan=colspan, sticky='ew', padx=5, pady=5   )
        style = ttk.Style()
        custom_font = tkFont.Font(**LAYOUT['fonts']['sliders']['params'])
        style.configure(LAYOUT['fonts']['sliders']['name'], font=custom_font)
        
        label = ttk.Label(frame, text=param['label'][5:], font=custom_font)
        label.pack(side='top', anchor='w')
        
        var_type = tk.IntVar if param['type'] == 'int' else tk.DoubleVar
        var = var_type(value=param['default'])
        slider = ttk.Scale(frame, from_=param['range'][0], length=param['length'],
                           to=param['range'][1], orient='horizontal', variable=var)
        slider.pack()
        
        setattr(self, param['name'], var)

    def create_button(self, parent, button):
        btn = ttk.Button(parent, text=button['label'], command=lambda: self.on_button_click(button['name']))
        btn.grid(row=button['row_col'][0], 
                 column=button['row_col'][1], padx=5, pady=5)
        
    def on_button_click(self, button_name):
        if button_name == 'save_divider_init':
            self.save_divider_initialization()
        elif button_name == 'apply':
            self.apply_hough_transform()
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
        
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk
        
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