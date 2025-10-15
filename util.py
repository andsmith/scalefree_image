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

def fade(image, alpha):
    """ Fade an image towards white by factor alpha (0.0 = white, 1.0 = original image)
    image: HxWxC image
    alpha: fading factor
    returns: faded image
    """
    faded = (image.astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return faded



def poly_expand_features(xy, order):
    """
    Given an Nx2 array of x,y positions, return an NxM array of polynomial features up to self.order
    where M = (n+1)(n+2)/2 is the number of terms in a 2D polynomial of order n.
    :param xy: Nx2 array of x,y positions
    :param order: maximum polynomial order
    :return: NxM array of polynomial features
    """
    n = order
    N = xy.shape[0]
    features = [np.ones((N,1), dtype=xy.dtype)]  # bias term
    for i in range(1, n+1):
        for j in range(i+1):
            x_term = xy[:,0]**(i-j)
            y_term = xy[:,1]**j
            features.append((x_term * y_term).reshape(N,1))
    return np.hstack(features)


def test_poly_expand_features():
    # Test the poly_expand_features function
    xy = np.random.rand(100,2).astype(np.float32)*2.0 - 1.0
    x, y = xy[:,0].reshape(-1, 1), xy[:,1].reshape(-1, 1)
    ones = np.ones((xy.shape[0], 1), dtype=np.float32)
    xy_poly1 = np.concatenate([ones, x, y], axis=1)
    xy_poly2 = np.concatenate([xy_poly1, x**2, x*y, y**2], axis=1)
    xy_poly3 = np.concatenate([xy_poly2, x**3, x**2*y, x*y**2, y**3], axis=1)
    xy_poly4 = np.concatenate([xy_poly3, x**4, x**3*y, x**2*y**2, x*y**3, y**4], axis=1)

    def _test_order(order, expected):
        result = poly_expand_features(xy, order)
        assert result.shape == expected.shape, f"Shape mismatch for order {order}: got {result.shape}, expected {expected.shape}"
        assert np.allclose(result, expected), f"Value mismatch for order {order}"
    
    _test_order(1, xy_poly1)    
    _test_order(2, xy_poly2)
    _test_order(3, xy_poly3)
    _test_order(4, xy_poly4)
    
    
    print("All tests passed for poly_expand_features.")

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
        #image[y_text,x_text-10:] = 0
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
        

def captioned_image(img, caption, caption_height_px=30, caption_pad_xy=(10, 5), txt_color=(255,255,255), bkg_color=(0,0,0),
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
    
    
def make_central_weights(img_size_wh, w_max, r_inner, r_outer, offsets_xy_rel, smooth=3):
    """
    Make a weight matrix that weights pixels near the center more heavily using a Gaussian falloff 
    :param img_size_wh: (width, height) of the image
    :param w_max: maximum weight at center
    :param r_inner: radius (in 0..1) of inner region with max weight
    :param r_outer: radius (in 0..1) of outer region where weight falls to 1.0
    :param offsets_xy_rel: (x_offset, y_offset) relative offsets of center (0.5, 0.5 = center)
    """
    w, h = img_size_wh
    x_offset, y_offset = (offsets_xy_rel[0]-0.5)*w, (offsets_xy_rel[1]-0.5)*h
    yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    cx, cy = (w-1)/2.0 + x_offset, (h-1)/2.0 + y_offset
    rad = np.sqrt((xv - cx)**2 + (yv - cy)**2)
    rad_inner_px = r_inner * max(w, h)
    rad_outer_px = r_outer * max(w, h)
    weights = np.ones_like(rad, dtype=np.float32)
    mask_inner = rad <= rad_inner_px
    mask_outer = rad >= rad_outer_px
    mask_middle = np.logical_and(rad > rad_inner_px, rad < rad_outer_px)
    weights[mask_inner] = w_max
    weights[mask_outer] = 1.0
    weights[mask_middle] = 1.0 + (w_max - 1.0) * (1.0 - (rad[mask_middle] - rad_inner_px) / (rad_outer_px - rad_inner_px))
    if smooth > 1:
        weights = cv2.GaussianBlur(weights, (smooth|1, smooth|1), 0)
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) * (w_max - 1.0) + 1.0
    weights = np.clip(weights, 1.0, w_max)
        
    return weights.astype(np.float32)   

def test_make_central_weights():
    shape  =  (352, 649)
    weight = 2.0
    rad_inner = .3
    rad_outer = .35
    x_y_offset_rel = (0.45,0.3)
    
    fig, axes = plt.subplots(2, 1, figsize=(5, 7))
    axes = axes.flatten()
    w = make_central_weights(shape[::-1], w_max=weight, r_inner=rad_inner, r_outer=rad_outer, offsets_xy_rel=x_y_offset_rel, smooth=5)

    ax_image = axes[0]
    ax_cross_section = axes[1]  
                            
    
    # show contour lines at 10% intervals
    num_intervals = 5
    levels = [w.min() + i * ((w.max() - w.min()) / num_intervals) for i in range(num_intervals + 1)]
    ax_image.contour(w, levels=levels, cmap='viridis', linewidths=0.5)
    ax_image.set_title("Weights (contours at 20%% increments)", fontsize=10)
    
    cross_data = w[w.shape[0]//2,:]
    ax_cross_section.plot(cross_data, color='black')
    #ax_cross_section.set_title("Cross-section\nshape=%s, rad_rel=%.2f" % (shape, rad_rel), fontsize=10)
    #ax_cross_section.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_captioned_image():
    
    test_frame = cv2.imread('movies\\washington_linear_8d_10h_cycle-00000010.png')
    caption = ['This is a test caption', 'Second line of caption']
    frame = captioned_image(test_frame, caption, caption_height_px=50, line_spacing=2.0,caption_pad_xy=(10, 5), txt_color=(255,255,255), bkg_color=(50,50,50))

    cv2.imshow("Test Captioned Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pairwise_hamming(a, b, n_bits=64):
    """
    Compute pairwise Hamming distances between two arrays of integers.

    Parameters:
        a (np.ndarray): shape (N,), array of integers
        b (np.ndarray): shape (M,), array of integers
        n_bits (int): Only return this many bits (Least significant bits)

    Returns:
        np.ndarray: shape (N, M), Hamming distances
    """
    # Broadcast XOR between all pairs
    if n_bits < 64:
        mask = (1 << n_bits) - 1
        a = a & mask
        b = b & mask

    xor_vals = np.bitwise_xor(a[:, None], b[None, :]).astype(np.uint64)

    # View each 64-bit integer as 8 bytes
    bytes_view = xor_vals.view(np.uint8).reshape(xor_vals.shape + (8,))

    # Unpack each byte into 8 bits → shape (N, M, 8, 8)
    bits = np.unpackbits(bytes_view, axis=-1)

    # Count 1s across the last two axes → shape (N, M)
    return bits.sum(axis=(-1))

def test_pairwise_hamming():
    # Example
    a = np.array([1, 2, 3], dtype=np.uint64)  # 001, 010, 011
    b = np.array([0, 7], dtype=np.uint64)     # 000, 111
    print("a:", a)
    print("b:", b)
    dist = pairwise_hamming(a, b)
    print("Hamming distances:\n", dist)

def find_boundary_pixels(mask):
    """
    True value indicates inside the region. 
    Return a mask for every pixel touching a non-region pixel.
    Don't forget the borders of the image, which some regions will touch.
    """
    h, w = mask.shape
    boundary_x = (mask[1:,:] != mask[:-1,:]) | \
                 (mask[:-1,:] != mask[1:,:]) 
    boundary_y = (mask[:,1:] != mask[:,:-1]) | \
                 (mask[:,:-1] != mask[:,1:])
    bm_temp = np.zeros_like(mask, dtype=bool)
    bm_temp[1:,:] |= boundary_x
    bm_temp[:-1,:] |= boundary_x
    bm_temp[:,1:] |= boundary_y
    bm_temp[:,:-1] |= boundary_y
    
    # Keep border True:
    # bm_temp[0,:] |= True
    # bm_temp[-1,:] |= True
    # bm_temp[:,0] |= True
    # bm_temp[:,-1] |= True
    
    # Keep border True if mask was true at that border location:
    # bm_temp[0,:] |= mask[0,:]
    # bm_temp[-1,:] |= mask[-1,:]
    # bm_temp[:,0] |= mask[:,0]
    # bm_temp[:,-1] |= mask[:,-1]
    
    print("Mean boundary mask value:", bm_temp.mean())
    return np.where(bm_temp)

def pixels_in_bbox(bbox, pixels_xy):
    """
    Return true if at least one of the pixels is in the bounding box.
    :param bbox: {'x':(x_min, y_min), 'y': (x_max, y_max)}
    :param pixels_xy: N x 2 array of pixel coordinates
    """
    print(pixels_xy.shape)
    x_min, y_min, x_max, y_max = bbox['x'][0], bbox['y'][0], bbox['x'][1], bbox['y'][1]
    inside = np.any((pixels_xy[:,0] >= x_min) & (pixels_xy[:,0] < x_max) &
                    (pixels_xy[:,1] >= y_min) & (pixels_xy[:,1] < y_max))
    return inside

def test_pixels_in_bbox():
    img_size = (100, 100)
    bbox =  {'x':(65, 95), 'y': (5, 95)}
    x_min, y_min, x_max, y_max = bbox['x'][0], bbox['y'][0], bbox['x'][1], bbox['y'][1] 
    n_tests = 10
    test_size = 5
    spread = 10
    points, results = [], []
    for _ in range(n_tests):
        center = np.random.rand(2) * img_size
        dpoints = np.random.randn(test_size, 2) * spread + center
        dpoints = np.clip(dpoints, 0, np.array(img_size)-1).astype(int)
        result = pixels_in_bbox(bbox, dpoints)
        points.append(dpoints)
        results.append(result)
        
    points = np.vstack(points)
    results = np.array(results)
    
    # plot bbox
    plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min],'-')
    
    # plot point sets in different colors, X for outside, O for inside
    for i in range(n_tests):
        pts = points[i*test_size:(i+1)*test_size]
        if results[i]:
            plt.plot(pts[:,0], pts[:,1], 'o')
        else:
            plt.plot(pts[:,0], pts[:,1], 'x')
            
    plt.xlim(0, img_size[0])
    plt.ylim(0, img_size[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Pixels in bbox test")
    plt.show()
    
def test_find_boundary_pixels():
    mask = np.zeros((20,20), dtype=bool)
    x, y = make_input_grid((mask.shape[0], mask.shape[1]), resolution=1.0, keep_aspect=True)
    c=0.23, -.4
    r = .3432
    circle = (x - c[0])**2 + (y - c[1])**2 < r**2
    mask |= circle
    mask[:4,:] = True
    mask[:,:5] = True
    # mask = ~ mask
    boundary = find_boundary_pixels(mask)
    
    img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # img[mask, :] = (155, 0,0)
    img[boundary] = np.array([0, 155, 0], dtype=np.uint8)
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Boundary Pixels")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #test_captioned_image()
    # test_add_text()
    # test_make_input_grid()
    # #test_make_central_weights()
    # test_poly_expand_features()
    # test_pairwise_hamming()
    # test_find_boundary_pixels()
    test_pixels_in_bbox()
    logging.info("All tests passed.")
