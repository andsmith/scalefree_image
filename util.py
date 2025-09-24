import matplotlib.pyplot as plt
import numpy as np

import cv2
import logging


def downscale_image(image, factor):
    if factor == 1.0:
        return image
    new_shape = (np.array(image.shape[:2]) / factor).astype(int)
    logging.info("Downsampling (factor = %.1f) image from %s to %s" % (factor, image.shape, new_shape))
    return cv2.resize(image, new_shape[::-1], interpolation=cv2.INTER_AREA)


def make_input_grid(img_shape=None, resolution=1.0, border=0.0, keep_aspect=True):
    """ Make a grid of input coordinates in [-1,1]x[-1,1]
    img_shape: (height, width, channels)
    resolution: scaling factor for number of points (1.0 = one point per pixel)
    border: extra border around [-1,1]x[-1,1] (in input coordinates)
    keep_aspect: if True, keep the aspect ratio of the input image, padding with extra border as needed
    returns: (x_coords, y_coords) meshgrid arrays
    """
    scale = border + 1.0

    h, w = img_shape[0]*resolution, img_shape[1]*resolution
    xs = (((np.arange(w, dtype=np.float32)+.5) / float(w) * 2.0 - 1.0) * scale)
    ys = (((np.arange(h, dtype=np.float32)+.5) / float(h) * 2.0 - 1.0) * scale)

    xv, yv = np.meshgrid(xs, ys)
    if keep_aspect:
        aspect = img_shape[1] / img_shape[0]
        if aspect > 1.0:
            # wide image, pad y
            yv = yv / aspect
        else:
            # tall image, pad x
            xv = xv * aspect
    return xv, yv


def test_make_input_grid():
    x, y = make_input_grid((1000, 2000), resolution=0.5, margin=0.1)
    x, y = x.flatten(), y.flatten()
    grid = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    print("grid shape:", grid.shape)
    print("grid min/max:", grid.min(axis=0), grid.max(axis=0))
    print("grid mean:", grid.mean(axis=0))
    assert np.all(grid[:, 0] >= -1.1) and np.all(grid[:, 0] <= 1.1)
    assert np.all(grid[:, 1] >= -1.1) and np.all(grid[:, 1] <= 1.1)
    assert np.isclose(np.mean(grid[:, 0]), 0.0, atol=0.01)
    assert np.isclose(np.mean(grid[:, 1]), 0.0, atol=0.01)

def add_text(image, text_lines, bbox, line_spacing = 1.5, max_font_scale=3.0, min_font_scale=0.1, margin_xy=(15,5),
             font_face=cv2.FONT_HERSHEY_SIMPLEX, font_thickness=1, color=(128,128,128), justify='center',v_spread=False):
    """ Add text to an image within a bounding box
    image: HxWx3 BGR image
    text_lines: list of strings, one per line
    bbox: {'x': (x-min, x-max), 'y': (y-min, y-max)} bounding box in image coordinates
    font: dictionary of font parameters for cv2.putText()
    line_spacing: spacing between lines, as a multiple of font height
    """
    (x0, x1), (y0, y1) = bbox['x'], bbox['y']
    font_thickness = font_thickness
    font_face = font_face

    font_scale = 3.0
    width = x1 - x0 - 2*margin_xy[0]
    height = y1 - y0 - 2*margin_xy[1]
    test_text_width = width *2.0
    test_text_height = height * 2.0

    def _calc_y_spacing(font_scale, include_descenders = True):
        """
        calculate height of each line of text, space everything out vertically.
        include_descenders: if True, include descender height in line height calculation
        returns: array of line heights, vertical spacing between lines where:
             n_lines * line_height + (n_lines-1)*spacing  + 1 descender_height = height
            (the descender height is added to account for the last line's descender,
            or is added to each if include_descenders is True)
        """  
        line_heights = []
        y_text=0
        for line in text_lines:
            (w,h), b = cv2.getTextSize(line, font_face, font_scale, font_thickness)
            b = b if include_descenders else 0
            line_heights.append(h + b)
        spacing = int((h+b) * (line_spacing-1))
        return line_heights, spacing

    def _calc_widths(font_scale):
        widths = []
        for line in text_lines:
            (w,h), b = cv2.getTextSize(line, font_face, font_scale, font_thickness)
            widths.append(w)
        return widths

    while (test_text_width > width or test_text_height > height) and font_scale > min_font_scale:
        font_scale -= 0.01
        widths = _calc_widths(font_scale)
        test_text_width = max(widths)
        line_heights, v_spacing = _calc_y_spacing(font_scale)
        test_text_height = sum(line_heights) + v_spacing * (len(text_lines)-1)

    font_scale = np.clip(font_scale, min_font_scale, max_font_scale)
    widths = _calc_widths(font_scale)
    line_heights, v_spacing = _calc_y_spacing(font_scale)

    extra_y_space = height - (sum(line_heights) + v_spacing * (len(text_lines)-1))
    # recompute y so text is vertically centered.  At top is the top of the first line, bottom the baseline of last line.
    y_start = y0 + extra_y_space // 2
    
    for i, line in enumerate(text_lines):
        y_text = y_start + line_heights[i]   # add descender?

        if justify == 'left':
            x_text = x0
        elif justify == 'right':
            x_text = x1 - widths[i]
        elif justify == 'center':
            x_text = x0 + (x1 - x0 - widths[i]) // 2  # center text horizontally
        else:
            raise ValueError("Unknown justify option: %s" % justify)
        y_start = y_text + v_spacing
        cv2.putText(image, line, (x_text, y_text), font_face, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    #draw_bbox(image, bbox, color=(255, 255, 0), thickness=4)


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=1):
    """ Draw a bounding box on an image
    image: HxWx3 BGR image
    bbox: {'x': (x-min, x-max), 'y': (y-min, y-max)} bounding box in image coordinates
    color: BGR color tuple
    thickness: line thickness
    """
    (x0, x1), (y0, y1) = bbox['x'], bbox['y']
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
        

def captioned_frame(img, caption, caption_height_px=30, caption_pad_xy=(10, 5), txt_color=(255,255,255), bkg_color=(0,0,0),
                    justify='center', line_spacing=1.5, font_face=cv2.FONT_HERSHEY_SIMPLEX, **kwargs):
    caption_h = caption_height_px
    pad_x, pad_y = caption_pad_xy
    img_cap = np.zeros((img.shape[0]+caption_h, img.shape[1], 3), dtype=np.uint8)
    img_cap[:img.shape[0], :, :] = img
    img_cap[img.shape[0]:, :, :] = bkg_color
    img_cap[:img.shape[0], :img.shape[1], :] = img
    bbox = {'x': (pad_x, img.shape[1]-pad_x),
            'y': (img.shape[0]+int(pad_y)-2, img_cap.shape[0]-2)}
    add_text(img_cap, caption, bbox,font_face=font_face, justify=justify, color=txt_color, line_spacing=line_spacing, **kwargs)
    # draw_bbox(img_cap, bbox, color=(0, 255, 0), thickness=1)
    return img_cap

def test_add_text():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    bbox = {'x': (50, 350), 'y': (50, 150)}
    text_lines = ["This is a test", "of the add_text function.", "It should center text", "within the bounding box."]
    add_text(img, text_lines, bbox, line_spacing=1.5)
    draw_bbox(img, bbox, color=(0, 255, 0), thickness=1)
    cv2.imshow("Test Add Text", img)
    cv2.waitKey(0)
    img *=0
    
    text_lines = ['This is another test that is ','left justified [and 2 lines].']
    add_text(img, text_lines, bbox, line_spacing=3.5, justify='left')
    draw_bbox(img, bbox, color=(0, 255, 0), thickness=1)
    cv2.imshow("Test Add Text Left Justified", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def make_central_weights(img_size_wh, max_weight=10.0, rad_rel=0.5, offsets_rel=(0.5,0.5), flatness = 0.0):
    """
    Make a weight matrix that weights pixels near the center more heavily using a Gaussian falloff 
    :param img_size_wh: (width, height) of the image
    :param max_weight: maximum weight at the center
    :param rad_rel: radius (relative to half the image diagonal) at which the weight falls to 50% of max_weight
    :param offsets_rel: (x,y) offsets of the center relative to image size (0.5,0.5 = center of image)
    :param flatness: The values above this fraction of max_weight are clipped, then everything
    is scaled to [1.0, max_weight].  (0.0 = no clipping, 1.0 = all weights are max_weight)
    :return: weight matrix of shape (height, width)
    """
    w, h = img_size_wh
    x_offset, y_offset = (offsets_rel[0]-0.5)*w, (offsets_rel[1]-0.5)*h
    yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    cx, cy = (w-1)/2.0 + x_offset, (h-1)/2.0 + y_offset
    rad = np.sqrt((xv - cx)**2 + (yv - cy)**2)
    max_rad = np.sqrt((w/2.0)**2 + (h/2.0)**2)
    sigma = rad_rel * max_rad / np.sqrt(2.0 * np.log(2.0))  # so that weight is half max_weight at rad_rel * max_rad
    weights = 1.0 + (max_weight - 1.0) * np.exp(-0.5 * (rad/sigma)**2)
    weights = weights - np.min(weights) 
    weights = weights / np.max(weights) * (max_weight-1.0) 
    if flatness > 0.0:
        nonflatness = 1.0 - flatness
        clip_val = weights.max() * nonflatness
        print("Clipping weights above %.3f (%.1f%% of max)" % (clip_val, nonflatness*100.0))
        weights = np.clip(weights, 1.0, clip_val)
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) * (max_weight - 1.0) + 1.0
    return weights.astype(np.float32)   

def test_make_central_weights():
    shapes = [(100, 100), (200, 140), (130, 200)]
    weight = 5.0
    rad_rels = [0.1, 0.3, 0.5, .75, .9]
    flatness = 0.5
    fig, axes = plt.subplots(len(shapes)*2, len(rad_rels), figsize=(16, 10))
    for i, shape in enumerate(shapes):
        for j, rad_rel in enumerate(rad_rels):
            w = make_central_weights(shape, max_weight=weight, rad_rel=rad_rel, flatness=flatness)

            ax_image = axes[i*2, j]
            ax_cross_section = axes[i*2+1, j]
            
            # show contour lines at 10% intervals
            num_intervals = 5
            levels = [w.min() + i * ((w.max() - w.min()) / num_intervals) for i in range(num_intervals + 1)]
            ax_image.contour(w, levels=levels, cmap='viridis', linewidths=0.5)
            ax_image.set_title("Weights (contours at 20%% increments)", fontsize=10)
            
            cross_data = w[w.shape[0]//2,:]
            ax_cross_section.plot(cross_data, color='black')
            ax_cross_section.set_title("Cross-section\nshape=%s, rad_rel=%.2f" % (shape, rad_rel), fontsize=10)
            #ax_cross_section.axis('off')
            
    plt.tight_layout()
    plt.show()

def test_captioned_frame():
    
    test_frame = cv2.imread('movies\\washington_linear_8d_10h_cycle-00000010.png')
    caption = ['This is a test caption', 'Second line of caption']
    frame = captioned_frame(test_frame, caption, caption_height_px=50, line_spacing=2.0,caption_pad_xy=(10, 5), txt_color=(255,255,255), bkg_color=(50,50,50))

    cv2.imshow("Test Captioned Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def interpolate_weights(locs, grid, shape):
    """
    Interpolate weights defined on a grid to arbitrary locations using bilinear interpolation
    :param locs: Nx2 array of (x,y) locations in [0,1]x[0,1]
    :param grid: HxW array of weights defined on a regular grid
    :param shape: (width, height) of the image that locs correspond to
    :return: N array of interpolated weights
    """
    h, w = grid.shape
    img_w, img_h = shape
    xs = locs[:, 0] * (img_w - 1)
    ys = locs[:, 1] * (img_h - 1)

    # map image coordinates to grid coordinates
    grid_xs = xs / (img_w - 1) * (w - 1)
    grid_ys = ys / (img_h - 1) * (h - 1)

    x0 = np.floor(grid_xs).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(grid_ys).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)

    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wa = (x1 - grid_xs) * (y1 - grid_ys)
    wb = (grid_xs - x0) * (y1 - grid_ys)
    wc = (x1 - grid_xs) * (grid_ys - y0)
    wd = (grid_xs - x0) * (grid_ys - y0)

    weights = wa * grid[y0, x0] + wb * grid[y0, x1] + wc * grid[y1, x0] + wd * grid[y1, x1]
    return weights

def test_interpolate_weights():
    grid = make_central_weights((10, 10), max_weight=5.0, rad_rel=0.5, flatness=0.2)
    shape = (200, 200)
    x, y = make_input_grid((shape[1], shape[0]), resolution=1.0, border=0.0, keep_aspect=True)
    locs = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    locs = (locs + 1.0) / 2.0  # map from [-1,1] to [0,1]
    weights = interpolate_weights(locs, grid, shape)

    weights_img = weights.reshape((shape[1], shape[0]))
    cv2.imshow("Interpolated Weights", weights_img / np.max(weights_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def calc_grid_size(n_units, aspect_ratio=1.0):
    """
    Calculate a good grid size (rows, cols) for n_units with the given aspect ratio (width/height)
    :param n_units: number of units to arrange in a grid
    :param aspect_ratio: desired aspect ratio (width/height) of the grid
    :return: (rows, cols)
    """
    cols = int(np.ceil(np.sqrt(n_units * aspect_ratio)))
    rows = int(np.ceil(n_units / cols))
    return rows, cols

def test_calc_grid_size():
    test_cases = [
        (10, 1.0),
        (20, 1.0),
        (50, 1.0),
        (100, 1.0),
        (10, 2.0),
        (20, 2.0),
        (50, 2.0),
        (100, 2.0),
        (10, 0.5),
        (20, 0.5),
        (50, 0.5),
        (100, 0.5),
    ]
    for n_units, aspect_ratio in test_cases:
        rows, cols = calc_grid_size(n_units, aspect_ratio)
        print(f"n_units={n_units} =?= {rows * cols}, aspect_ratio={aspect_ratio:.2f} => rows={rows}, cols={cols}, aspect diff={aspect_ratio - cols/rows:.2f}")
        assert rows * cols >= n_units


def sample_image(image, n_samples,weights=None):
    """
    Don't just sample uniformly, over a grid of points with approx n_samples (keeping aspect ratio), and 
    with noise added so each point can be anywhere within it's grid cell
    :param image: HxWx3 BGR image
    :param n_samples: approximate number of samples to take
    :param weights: optional array of weights to sample at the same coordinates (must match image size)
    """
    aspect = image.shape[1] / image.shape[0]
    rows, cols = calc_grid_size(n_samples, aspect)
    
    
    x, y = np.meshgrid(np.linspace(0,image.shape[1], cols, dtype=int, endpoint=False) ,
                       np.linspace(0,image.shape[0], rows, dtype=int, endpoint=False)) 
    x_scatter = image.shape[1] / cols
    y_scatter = image.shape[0] / rows
    
    x += np.random.randint(0, x_scatter, size=x.shape)
    y += np.random.randint(0, y_scatter, size=y.shape)
    red_outputs = image[y, x, 0] / 255.0
    green_outputs = image[y, x, 1] / 255.0
    blue_outputs = image[y, x, 2] / 255.0

    
    # Now compute input coordinates
    
    if aspect > 1.0:
        # wide image, shrink y, x is in [-1, 1]
        x_span = -1.0, 1.0
        y_span = -1.0/aspect, 1.0/aspect
    else:
        # tall image, shrink x, y is in [-1, 1]
        x_span = -aspect, aspect
        y_span = -1.0, 1.0
        
    input_x =  x / (image.shape[1]-1) * (x_span[1] - x_span[0]) + x_span[0]
    input_y =  y / (image.shape[0]-1) * (y_span[1] - y_span[0]) + y_span[0]


    input = np.hstack((input_x.reshape(-1,1), input_y.reshape(-1,1)))
    output = np.hstack((red_outputs.reshape(-1,1), green_outputs.reshape(-1,1), blue_outputs.reshape(-1,1)))
    if weights is not None:
        weights = weights[y, x].reshape(-1,1)
    else:
        weights = None
    print("Input shape:", input.shape, "Output shape:", output.shape)
    return input, output, weights

def make_test_image(box_size = 10, n_boxes = 10):
    # Vertical si red fading to black, horizontal is green fading to black
    # diagonal is blue fading to black
    c_val = np.array([i for i in np.linspace(0, 255, n_boxes).astype(np.uint8)])
    c_val = np.repeat(c_val, box_size)
    red = np.tile(c_val, (box_size*n_boxes,1))
    green = np.tile(c_val[::-1].reshape(-1,1), (1,box_size*n_boxes))
    blue =0 * red
    for i in range(n_boxes):
        for j in range(n_boxes):
            color = int(((i+j)/18.0)*255)
            blue[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size] = color

    image = np.dstack((red, green, blue))
    return image


def test_sample_image():
    image = make_test_image(n_boxes=3,box_size = 300)
    # test image is 10x10 colored boxes
    inputs, outputs = [], []
    for _ in range(10):
        input_xy, output_rgb = sample_image(image, 150)
        inputs.append(input_xy)
        outputs.append(output_rgb)
    in_x = np.array(inputs).reshape(-1,2)
    in_y = np.array(outputs).reshape(-1,3)  
    for input_xy, output_rgb in zip(inputs, outputs):
        plt.scatter(in_x[:,0], in_x[:,1], c=in_y, s=15)
    plt.axis('equal')
    plt.title("Colors should be all linearly separable.")
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #test_captioned_frame()
    # test_add_text()
    # test_make_input_grid()
    #test_make_central_weights()
    test_interpolate_weights()
    # test_calc_grid_size()
    #test_sample_image()
    
    logging.info("All tests passed.")
