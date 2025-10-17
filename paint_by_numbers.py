"""
Paint-by-numbers:

     * The interval [-1,1]x[-1,1] is divided into 2 regions (1 and 0) by N ordered dividers (circles and/or lines).
     * This results in that interval being cut into R regions, where R is O(N^2).
     * Each region can be coded as an N-bit binary number indicating which side of the N dividers it is on.
     * The color look up table (LUT) maps codes to the average color of pixels for all pixels in the target image with
         that code, i.e. from the same region.

   Define functions:
   
       make_lut(image, divider_masks):  Find the regions with the same code (divider responses), to each associate the 
         average color of the pixels in the image that have that code.  Return a dict mapping code (tuple of uint64s) to color (3-vector).
         
   The process of getting RGB values from X, Y locations is then:
      1. Evaluate the N dividers at the X,Y locations to get an N-bit code for each location.
      2. Look up the code in the LUT to get the color. 
"""
import logging
import cv2
import numpy as np
from synth_image import TestImageMaker
import pickle
import matplotlib.pyplot as plt
from util import make_input_grid, pairwise_hamming,find_boundary_pixels,pixels_in_bbox

from abc import ABC, abstractmethod


class DivMask(object):
    def __init__(self, divider, shape_hw):
        self.inside = divider.make_divmap(shape_hw)
        self.outside = np.logical_not(self.inside)
        self.border_px = find_boundary_pixels(self.inside)


    def find_regions_crossed(self, regions):
        """
        Find subset of 'regions' that are crossed by the given divider mask (i.e. might have pixels on both sides of the divider).
        Returns:  List crossed, where crossed[i] is the index of the region that is crossed by the divider, or None if not crossed.
                    List sides, where sides[i] is a tuple (side1_mask, side2_mask) of boolean masks for the two sides of the divider
                        within region i, or (None, None) if not crossed.
        """
        crossed = []
        sides = []
        n_skipped = 0
        for r_i, region in enumerate(regions):
            if not pixels_in_bbox(region.bbox, np.array(self.border_px)):
                # if the divider does not intersect the bounding box of the region, skip it
                crossed.append(None)
                sides.append((None, None))
                n_skipped += 1
                continue
            # if there is an intersection, cut the regions (return 2 regions)
            side_region1, side_region2 = region.cut( self)
            if side_region1 is not None:
                crossed.append(r_i)
                sides.append((side_region1, side_region2))
            else:
                crossed.append(None)
                sides.append((None, None))
        logging.info(f"Divider crossed {len(crossed) - n_skipped} regions, skipped {n_skipped} regions that did not intersect the divider bounding box.")
        return crossed, sides
    
class Region(object):
    def __init__(self, mask, bbox=None):
        self.mask, self.bbox = self.prune_mask(mask, old_bbox=bbox)

    @staticmethod
    def prune_mask(mask, old_bbox=None):
        """
        find the smallest bounding box containing all True values in the mask, 
        
        :param mask: boolean array of shape (H, W)
        :param old_bbox: optional (y0, x0, y1, x1), if given, it's offset will be added to the retuned bbox so they are
            both with respect to the same origin.
            
        return dict {'mask': mask[bbox],  # boolean array, the mask without any FALSE rows/cols on the margins
                    'offset': (y0, x0),  # the offset of the returned mask within the original mask
                    }
        """
        ys, xs = np.where(mask)
        if len(ys) == 0 or len(xs) == 0:
            return np.zeros((0,0), dtype=bool)
        y0, y1 = np.min(ys), np.max(ys) + 1
        x0, x1 = np.min(xs), np.max(xs) + 1
        pruned = mask[y0:y1, x0:x1].reshape((y1 - y0, x1 - x0))
        if old_bbox is not None:
            y0 += old_bbox['y'][0]
            x0 += old_bbox['x'][0]
            y1 += old_bbox['y'][0]
            x1 += old_bbox['x'][0]
        return pruned,  {'y':(y0, y1), 'x':(x0, x1)}


    def cut(self, div_mask):
        """
        Returns:  side_region1, side_region2, where each is a dict with keys:
                    'mask': boolean array of shape (h, w) indicating the pixels in that side of the region
                    'bbox': (x_min, y_min, x_max, y_max) bounding box of the region
                or None if the region is not crossed by the divider.
        """
        region_wh = self.mask.shape
        div_mask_roi = div_mask.inside[self.bbox['y'][0]:self.bbox['y'][1],
                                       self.bbox['x'][0]:self.bbox['x'][1]].reshape(region_wh)
        side1_mask = np.logical_and(self.mask, div_mask_roi)
        side2_mask = np.logical_and(self.mask, np.logical_not(div_mask_roi))
        side_region1 = Region(side1_mask, bbox=self.bbox) if np.sum(side1_mask) > 0 else None
        side_region2 = Region(side2_mask, bbox=self.bbox) if np.sum(side2_mask) > 0 else None
        return side_region1, side_region2

    

class Divider(ABC):
    def __init__(self, params):
        # params is 3 floats (x, y, angle) for line, (x,, y, log-radius) for circle
        self.params = params
        
    @abstractmethod
    def eval(self, x, y):
        """
        Evaluate the divider at the given x,y locations.
        :param x:  np.array of shape (n_points,) with x coordinates
        :param y:  np.array of shape (n_points,) with y coordinates
        :return:  np.array of shape (n_points,) with True or False depending on which side of the divider
        """
        pass
    
    @abstractmethod
    def make_rand():
        """
        Make a random divider within the unit square.
        :return:  Divider object
        """
        pass
    
    def make_divmap(self, shape):
        """
        Make a boolean mask of the given shape, where True indicates the pixel is on one side of the divider.
        :param shape:  tuple (H, W) indicating the shape of the mask to create
        :return:  boolean array of shape (H, W)
        """
        x, y = make_input_grid(shape, resolution=1.0, keep_aspect=True)
        inside = self.eval(x, y)
        return inside
    
    @abstractmethod
    def plot(self, ax, xlim, ylim):
        pass

class LineDivider(Divider):
    def eval(self, x, y):
        #x,y = y, x
        angle = self.params[2]
        center = self.params[:2]
        vec = np.stack([x,y], axis=-1) - center
        unit = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        line_vec = np.array([np.cos(angle), np.sin(angle)])
        dists = np.sum(unit * line_vec, axis=-1)
        return dists >= 0
    
    def make_rand():
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        angle = np.random.uniform(0, 2.0 * np.pi)
        return LineDivider(np.array([x, y, angle], dtype=np.float32))
    
    def plot(self, ax, xlim, ylim): 

        angle = np.pi/2.0 - self.params[2] 
        center = self.params[:2]
        
        unit = np.array([np.cos(angle), np.sin(angle)])
        center[1] = -center[1]
        t = [-10.0, 10.0]
        line_pts = np.array([center + unit * ti for ti in t])
        
        
        ax.plot(line_pts[:,0], line_pts[:,1], 'r-')
        ax.plot(center[0], center[1], 'ro')
    
class CircleDivider(Divider):
    def eval(self, x, y):
        center = self.params[:2]
        radius = np.exp(self.params[2])
        disp = np.stack([x, y], axis=-1) - center
        dist_sq = np.sum(disp**2, axis=-1)
        return dist_sq <= radius**2  # inside circle is True
    
    def make_rand():
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        log_radius = np.random.uniform(np.log(0.05), np.log(0.5))
        return CircleDivider(np.array([x, y, log_radius], dtype=np.float32))
    
    def plot(self, ax, xlim, ylim): 
        center = self.params[:2]
        center[1] = -center[1] #+ res[1]/2.0  

        radius = np.exp(self.params[2])
        circle = plt.Circle((center[0], center[1]), radius, color='b', fill=False)
        ax.add_artist(circle)
        ax.plot(center[0], center[1], 'bo')
    
    
    
def dividers_from_model(filename):
    """
    NNetImage model state files have a top-level dict with keys:
        - 'weights', a list of numpy arrays.
        - 'n_div', a dict with {'linear': int, 'circular': int, 'normal': int} 
        
    Divider weights are in groups of 3 arrays (Center, Angle/radius, _ ).
    Divider weights are first in the 'weights' list.  
    If both circular and linear dividers are used, circular dividers come first.
    Normal (sigmoid) dividers are unimplemented in this module.
    """
    with open(filename, 'rb') as f:
        state = pickle.load(f)  
    n_div = state['n_div']
    weights = state['weights']
    dividers = []
    w_i = 0
    # print("Loaded weights:", [w.shape for w in weights])
    if n_div['circular']>0:
        centers = weights[w_i]
        log_radii = weights[w_i+1].reshape(-1,1)
        param_arr = np.concatenate([centers, log_radii], axis=1)
        dividers.extend([CircleDivider(param_arr[i]) for i in range(n_div['circular'])])
        w_i += 3 
    if n_div['linear']>0:
        centers = weights[w_i]
        angles = weights[w_i+1].reshape(-1,1)
        print(centers.shape,angles.shape)
        param_arr = np.concatenate([centers, angles], axis=1)
        
        dividers.extend([LineDivider(param_arr[i]) for i in range(n_div['linear'])])
        w_i += 3 
    logging.info(f"Loaded {len(dividers)} dividers from model {filename}: {n_div}")
    return dividers
        
        
        


class ColorEncoding(object):
    _ENCODING_TYPE = np.uint32
    _ENCODING_BITS = 8 * np.dtype(_ENCODING_TYPE).itemsize  # 64 bits
    
    def __init__(self, dividers):
        self.dividers = dividers  # list of Divider objects
        self._n_codewords = int(np.ceil(len(dividers) / self._ENCODING_BITS))
        self._LUT = None  # Dict mapping self._n_codewords-tuples to 3-vectors of uint8 colors
        logging.info(f"Initialized ColorEncoding with {len(dividers)} dividers, {self._n_codewords} codewords per code (type {self._ENCODING_TYPE} with {self._ENCODING_BITS} bits each).")
        
    def encode_xy_points(self, x, y):
        """
        Encode the given x,y points using the dividers to get their binary codes.
        
        :param x:  np.array of shape (... ) with x coordinates
        :param y:  np.array of shape (...,) with y coordinates
        :return:  np.array of shape (..., n_bytes) with the binary codes as uint8 arrays
        """
        data_shape = x.shape
        n_points = x.shape[0]
        n_dividers = len(self.dividers)
        codes = np.zeros(data_shape + (self._n_codewords,), dtype=self._ENCODING_TYPE)
        
        for bit_place, divider in enumerate(self.dividers):
            mask = divider.eval(x, y)  # boolean array of shape (n_points,)
            byte_index = bit_place // self._ENCODING_BITS
            bit_index = bit_place % self._ENCODING_BITS
            codes[...,mask, byte_index] |= (1 << bit_index)
        
        return codes
    
    def get_colors(self, codes):
        """
        Get the colors for the given codes from the learned color table.
        
        :param codes:  np.array of shape (n_points, n_bytes) with the binary codes as uint8 arrays
        :return:  np.array of shape (n_points, 3) with the RGB colors as uint8 arrays
        """
        if not hasattr(self, '_codes') or not hasattr(self, '_colors'):
            raise ValueError("ColorEncoding has not been trained yet. Call train_image() first.")
        data_shape = codes.shape[:-1]
        n_points = np.prod(data_shape)
        codes_flat = codes.reshape(-1, self._n_codewords)
        colors_flat = np.zeros((n_points, 3), dtype=np.uint8)
        
        unmatched = []
        for i in range(n_points):
            code = tuple(codes_flat[i])
            if code in self._LUT:
                colors_flat[i] = self._LUT[code]
                
            else:
                unmatched.append(i)
        logging.info("Found {}/{} unmatched codes, doing approximate nearest neighbor search.".format(len(unmatched), n_points))
        
        closest_matches = approx_code_lookup(self._codes, codes_flat[unmatched], n_bits=len(self.dividers))
        for idx, match in zip(unmatched, closest_matches):
            colors_flat[idx] = self._colors[match]
        colors = colors_flat.reshape(data_shape + (3,))
        return colors

    def render_regions(self, regions, size_wh, ax=None):
        """
        Create a full size array, add each reagion mask to it with a different index.
        """
        w, h = size_wh
        region_img = np.zeros((h, w), dtype=np.int32) - 1
        for i, region in enumerate(regions):
            x_min, y_min, x_max, y_max = region.bbox['x'][0], region.bbox['y'][0], region.bbox['x'][1], region.bbox['y'][1]
            region_img[y_min:y_max, x_min:x_max][region.mask] = i


        aspect, xlim, ylim = get_aspect_and_lims(region_img.shape)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6,6))
        
        ax.imshow(region_img, cmap='tab20', extent = (xlim[0], xlim[1], ylim[0], ylim[1]))

        for divider in self.dividers:
            divider.plot(ax, xlim, ylim)
                
        # add colorbar
        cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(cmap='tab20'), ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(-0.5, len(regions), 1))
        ax.set_title(f"Regions: {len(regions)}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
    
    def train_image(self, target_image):
        h, w = target_image.shape[0], target_image.shape[1]

        regions = self._find_regions(h, w)
        self.render_regions(regions, (w,h))
        plt.show()

        self._codes, self._colors = self._optimize_colors(target_image, regions)
        self._LUT = {tuple(self._codes[i]): self._colors[i] for i in range(self._codes.shape[0])}

    def _find_regions(self, h, w):
        """
        Find the unique codes for the given image & masks, 
        find the average color for each code.
        
        Algorithm:  
            1. Maintain a set of regions (a list of region bit masks, IDs, etc.),  Initialize with 1 region containing all pixels.
            2.  For each mask, find the subset of regions it crosses (has pixels on both sides of the mask).
                For each such region, split it into two new regions, one for each side of the mask.
            3.  Remove the old region, add the new regions.
                    
        Implementation details:        
           codes are stored as uint8 arrays, so there will be ceil(N/8) bytes per code.  
        
        :param h, w: height and width of the target image
        :returns: self.codes, the R x D array of binary codes (as uint8), and 
             self.colors, the  R x 3 array of colors (as uint8), where code[i] 
                 is the code for color[i].
        """

        regions = [Region(mask=np.ones((h, w), dtype=bool),
                     bbox={'x':(0, w), 'y':(0, h)})]  # Initialize with the full image as one region

        def split_all_regions(div_mask):
            nonlocal regions
            crossed_region_inds, sides = div_mask.find_regions_crossed(regions)
            new_regions = []
            for r_i, region in enumerate(regions):
                if crossed_region_inds[r_i] is None:
                    new_regions.append(region)
                else:
                    if sides[r_i][0] is not None:
                        new_regions.append(sides[r_i][0])
                    if sides[r_i][1] is not None:
                        new_regions.append(sides[r_i][1])
            regions = new_regions
            return np.sum([c is not None for c in crossed_region_inds])

        for bit_place, divider in enumerate(self.dividers):
            mask = DivMask(divider, (h, w))
            n_crossed = split_all_regions(mask)
            print(f"Mask {bit_place} crossed {n_crossed} regions, now have {len(regions)} regions")

        print(f"Found {len(regions)} regions from {len(self.dividers)} dividers.")
        return regions  

    def _optimize_colors(self, target_image, regions):
        """        
        Optimize the colors for each region based on the target image.
        """
        h, w = target_image.shape[0], target_image.shape[1]
        colors = np.zeros((len(regions), 3), dtype=np.uint8)
        codes = np.zeros((len(regions), self._n_codewords), dtype=self._ENCODING_TYPE)
        print(f"\n\n\nOptimizing colors for {len(regions)} regions.")
        
        
        n_pixels = 0

        for i,region in enumerate(regions):
            print(region.mask.astype(int))
            print(region.bbox)
            import ipdb; ipdb.set_trace()
            
            
            pruned_mask, offset_yx = region.mask, (region.bbox['y'][0], region.bbox['x'][0])
            n_pixels += np.sum(pruned_mask)
            mask_h, mask_w = pruned_mask.shape
            target_region = target_image[offset_yx[0]:offset_yx[0]+mask_h, offset_yx[1]:offset_yx[1]+mask_w, :].reshape(mask_h, mask_w)
            region_pixels = target_region[pruned_mask]

            
            if region_pixels.shape[0] > 0:
                avg_color = np.mean(region_pixels, axis=0)
                colors[i] = np.clip(avg_color, 0, 255).astype(np.uint8)
                
                print(f"Optimizing color for region {i}: has {region_pixels.shape[0]} pixels, avg color {colors[i]}, pruned_mask shape {pruned_mask.shape}, offset {offset_yx}")
            else:
                print(f"WARNING: Region {i} has no pixels in the target image.")
                colors[i] = np.array([0, 0, 0], dtype=np.uint8)  # default to black if no pixels
            
            for bit_place, divider in enumerate(self.dividers):
                div_mask = divider.make_mask_set((h, w))
                div_region = div_mask[offset_yx[0]:offset_yx[0]+mask_h, offset_yx[1]:offset_yx[1]+mask_w]
                side = np.logical_and(pruned_mask, div_region)
                if np.any(side):
                    byte_index = bit_place // self._ENCODING_BITS
                    bit_index = bit_place % self._ENCODING_BITS
                    codes[i, byte_index] |= (1 << bit_index)
                    
        return codes, colors

def _make_test_image():
    return cv2.imread('input/barn.png')[:,:,::-1]

def approx_code_lookup(codes, queries, n_bits, n_max_queries=1000):
    """
    return the code with the highest number of matching bits for each query
    :param codes:  np.array of shape (n_codes, n_codewords) with the binary codes as unit8 or int32s
    :param queries: np.array of shape (..., n_codewords) with the binary codes as the same dtype as codes
    :return: np.array of shape (queries.shape[0],) with the index of the closest code for each query
    """
    codeword_dtype = codes.dtype
    codebits = 8 * np.dtype(codeword_dtype).itemsize
    n_codes, n_code_words = codes.shape if len(codes.shape) == 2 else (codes.shape[0], 1)
    #Needs to be done in planes:
    data_shape = queries.shape[:-1] if n_code_words > 1 else queries.shape
    
    unique_queries = np.unique(queries.reshape(-1, n_code_words), axis=0)
    
    h_dist_unique = np.zeros((unique_queries.shape[0], n_codes), dtype=int)
    
    for w in range(n_code_words):
       num_bits = codebits if w < n_code_words - 1 else (n_bits - (n_code_words - 1) * codebits)
       h_dist_unique += pairwise_hamming(unique_queries[...,w],codes[..., w], num_bits)
    best_unique_code_inds = np.argmin(h_dist_unique, axis=-1)
    # Now fill them all in 
    best_code_inds = np.zeros(data_shape, dtype=int)
    for i, uq in enumerate(unique_queries):
        matches = np.all(queries == uq, axis=-1)
        best_code_inds[matches] = best_unique_code_inds[i]
    return best_code_inds
    
    
    

def test_approx_code_lookup():
    codes = np.array([0xFF,  # lowest 4 bits on
                      00, # all bits off
                      ], dtype=np.uint8)
    
    
    queries = np.arange(32, dtype=np.uint8)
    matches = approx_code_lookup(codes, queries, n_bits=5)
    print("Codes:\n", codes)
    print("Queries:\n", queries)
    print("Matches:\n", matches)
    print("Mean matches == code 1: ", np.mean(matches == 1))
    assert np.mean(matches == 1) == 0.5, "Half the queries should match code 1"

def get_aspect_and_lims(shape):
    h, w = shape[0], shape[1]
    aspect = w / h
    if aspect >= 1.0:
        xlim = (-1.0,1.0)
        ylim = (-1.0/aspect, 1.0/aspect)
    else:
        xlim = (-aspect, aspect)
        ylim = (-1.0, 1.0)
        
    return aspect, xlim, ylim

def test_make_LUT(image_size=(20,20), n_circles=0, n_lines=2):
    # dividers = [LineDivider.make_rand() for _ in range(n_lines)] + \
    #            [CircleDivider.make_rand() for _ in range(n_circles)]
               
    dividers = [LineDivider([0.1, -0.7, 0])]
                # LineDivider((0.0, 0.0, 0)),]
    
    dividers = dividers_from_model(r'test_test\SYNTH_bw_lines_test_model_2l_4c.pkl')
    #dividers = dividers_from_model(r'test_test_circles\SYNTH_bw_circles_test_model_2c_4c.pkl')
                                   # test_mix_3\SYNTH_mix_A_3_3_rand_model_5c-3l_15t_10c.pkl
    # dividers = dividers_from_model(r'test_mix_3\SYNTH_mix_A_3_3_rand_model_5c-3l_15t_10c.pkl')
    # dividers = dividers_from_model(r'test_barn\barn_model_16l_64c.pkl')
    image_maker = TestImageMaker(image_size_wh=image_size)   
    #image = image_maker.make_image('c_lines_5_rand')
    lines = {'centers': np.array([[0.0, 0.0001], [0.0, -0.0]]),
            'angles': np.array([0, np.pi/2])}
    #image = image_maker._synth_spec_image(lines=lines, is_color=False)
    image = image_maker.make_image('static_line_bw')
    
    # image = _make_test_image()
    # image = cv2.resize(image, (image_size[0], image_size[1]), interpolation=cv2.INTER_AREA)
    ce = ColorEncoding(dividers)
    print("Testing on input image:  %s  %s" % ('barn.png', str(image.shape)))
    
    ce.train_image(image)
    aspect, xlim, ylim = get_aspect_and_lims(image.shape)
    img_extent = (xlim[0], xlim[1], ylim[0], ylim[1])
    x_orig, y_orig = make_input_grid(image.shape[:2], resolution=1.0, keep_aspect=True)
    shape_big = np.array(image.shape[:2]) * 6
    x, y = make_input_grid(shape_big, resolution=1.0, keep_aspect=True)
    print(f"Encoding {x.size} points")
    codes = ce.encode_xy_points(x, y)
    print(f"Getting colors for {codes.shape[0]} codes (shape {codes.shape})")
    colors = ce.get_colors(codes)
    print(f"Got {colors.shape[0]} colors")
    colors_img = colors.reshape(shape_big[0], shape_big[1], 3)
    fig,ax=plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image, extent=img_extent)
    ax[0].plot(x_orig.flatten(), y_orig.flatten(), 'r.', markersize=1)
    ax[0].set_title("Original Image")
    ax[0].set_aspect('equal')
    ax[1].imshow(colors_img, extent=img_extent)
    ax[1].plot(x_orig.flatten(), y_orig.flatten(), 'r.', markersize=1)
    ax[1].set_title("Encoded Colors")
    ax[1].set_aspect('equal')   
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_approx_code_lookup()
    test_make_LUT()

    