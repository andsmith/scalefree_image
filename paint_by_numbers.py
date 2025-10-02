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

import matplotlib.pyplot as plt
from util import make_input_grid, pairwise_hamming

from abc import ABC, abstractmethod

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
    
    def make_mask(self, shape):
        """
        Make a boolean mask of the given shape, where True indicates the pixel is on one side of the divider.
        :param shape:  tuple (H, W) indicating the shape of the mask to create
        :return:  boolean array of shape (H, W)
        """
        x, y = make_input_grid(shape, resolution=1.0, keep_aspect=True)
        return self.eval(x, y)

class LineDivider(Divider):
    def eval(self, x, y):
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
    
        
    def train_image(self, target_image):
        h, w = target_image.shape[0], target_image.shape[1]
        regions = self._find_regions(h, w)
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
        #
        x, y = make_input_grid((h, w), resolution=1.0, keep_aspect=True)

        regions = [ np.ones((h, w), dtype=bool)]  # Initialize with the full image as one region

        
        def find_regions_crossed(div_mask):
            """
            Find all regions in 'regions' that are crossed by the given divider mask (i.e. have pixels on both sides of the divider).
            Returns:  List crossed, where crossed[i] is the index of the region that is crossed by the divider, or None if not crossed.
                      List sides, where sides[i] is a tuple (side1_mask, side2_mask) of boolean masks for the two sides of the divider
                          within region i, or (None, None) if not crossed.
            """
            crossed = []
            sides = []
            for r_i, region_mask in enumerate(regions):
                side1 = np.logical_and(region_mask, div_mask)
                side2 = np.logical_and(region_mask, np.logical_not(div_mask))
                if np.any(side1) and np.any(side2):
                    crossed.append(r_i)
                    sides.append((side1, side2))
                else:
                    crossed.append(None)
                    sides.append((None, None))
            return crossed, sides
        
        def split_regions(shape_mask):
            nonlocal regions
            crossed_region_inds, sides = find_regions_crossed(shape_mask)
            new_regions = []
            for r_i, region in enumerate(regions):
                if crossed_region_inds[r_i] is None:
                    new_regions.append(region)
                else:
                    new_regions.append(sides[r_i][0])
                    new_regions.append(sides[r_i][1])
            regions = new_regions

            return len(crossed_region_inds)

        for bit_place, divider in enumerate(self.dividers):
            mask = divider.make_mask((h, w))
            n_crossed = split_regions(mask)
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
        pruned = [prune_mask(reg) for reg in regions]

        for i,pruned_region in enumerate(pruned):
            pruned_mask, offset_yx = pruned_region['mask'], pruned_region['offset']
            n_pixels += np.sum(pruned_mask)
            mask_h, mask_w = pruned_mask.shape
            target_region = target_image[offset_yx[0]:offset_yx[0]+mask_h, offset_yx[1]:offset_yx[1]+mask_w, :]
            region_pixels = target_region[pruned_mask]

            
            if region_pixels.shape[0] > 0:
                avg_color = np.mean(region_pixels, axis=0)
                colors[i] = np.clip(avg_color, 0, 255).astype(np.uint8)
                
                print(f"Optimizing color for region {i}: has {region_pixels.shape[0]} pixels, avg color {colors[i]}, pruned_mask shape {pruned_mask.shape}, offset {offset_yx}")
            else:
                print(f"WARNING: Region {i} has no pixels in the target image.")
                colors[i] = np.array([0, 0, 0], dtype=np.uint8)  # default to black if no pixels
            
            for bit_place, divider in enumerate(self.dividers):
                div_mask = divider.make_mask((h, w))
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

def prune_mask(mask):
    """
    find the smallest bounding box containing all True values in the mask, 
    return dict {'mask': mask[bbox],  # boolean array, the mask without any FALSE rows/cols on the margins
                 'offset': (y0, x0),  # the offset of the returned mask within the original mask
                 }
    """
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        return np.zeros((0,0), dtype=bool)
    y0, y1 = np.min(ys), np.max(ys) + 1
    x0, x1 = np.min(xs), np.max(xs) + 1
    pruned = mask[y0:y1, x0:x1]
    return {'mask': pruned, 'offset': (y0, x0)}


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

def test_make_LUT(image_size=(32, 24), n_circles=50, n_lines=50):
    dividers = [LineDivider.make_rand() for _ in range(n_lines)] + \
               [CircleDivider.make_rand() for _ in range(n_circles)]
               
    # dividers = [LineDivider((0.0, 0.0, np.pi/2)),
    #             LineDivider((0.0, 0.0, 0)),]
    
    # image_maker = TestImageMaker(image_size_wh=image_size)   
    # image = image_maker.make_image('c_lines_5_rand')
    image = _make_test_image()
    # image = cv2.resize(image, (image_size[0], image_size[1]), interpolation=cv2.INTER_AREA)
    ce = ColorEncoding(dividers)
    
    
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

