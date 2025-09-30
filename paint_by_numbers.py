"""
Paint-by-numbers.  Given circles/lines that partition the plane, create a look-up table
for the regions (which combinations of dividers form the region, what its its average color)
"""
import numpy as np
import cv2
from util import make_input_grid


def eval_circles(self, x, y, circles):
    """
    Evaluate circular dividers at the given x,y locations.
    :param x:  np.array of shape (n_points,) with x coordinates
    :param y:  np.array of shape (n_points,) with y coordinates
    :param circles:  dict with keys 'centers' (np.array of shape (n_circles, 2)) and 'log-radii' (np.array of shape (n_circles,))
    :return:  np.array of shape (n_points, n_circles) with True or False depending on which side of the circle (inside/outside)
    """
    coords = np.vstack((x, y)).T
    disp = coords - circles['centers']
    
    dist_sq = np.sum(disp**2, axis=2)  # shape (n_points, n_circles)
    radii_sq = np.exp(2.0 * circles['log-radii'].flatten())  # shape (n_circles,)
    mask = np.where(dist_sq <= radii_sq,True, False)
    return mask

def eval_lines(self, x, y, lines):
    """
    Evaluate linear dividers at the given x,y locations.
    """
    coords = np.vstack((x, y)).T  # shape (n_points, 2)
    angles = lines['angles'].flatten()
    offsets = np.sum(lines['centers'] * np.vstack((np.cos(angles), np.sin(angles))).T, axis=1)
    n = np.vstack((np.cos(angles), np.sin(angles))).T  # shape (n_lines, 2)
    dists = np.dot(coords, n.T) - offsets  # shape (n_points, n_lines)
    mask = np.where(dists>=0 , True, False)
    return mask


def paint_by_code(self, codes):
    """
    Given codes (n_points, n_dividers) where each entry is +1 or -1 depending on which side of the divider
    the point is on, return the color of each point.
    
    codes:  np.array of shape (n_points, n_dividers), each entry +1 or -1
    returns:  np.array of shape (n_points, 3) with color values in [0,1]
    """
    h, w = codes.shape[0], codes.shape[1]
    colors = np.zeros((h, 3), dtype=np.float32)
    for i in range(h):
        code_tuple = tuple(codes[i,:].tolist())
        if code_tuple in self.color_LUT:
            colors[i,:] = self.color_LUT[code_tuple]
        else:
            colors[i,:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return colors

import numpy as np
import math

def bools_to_bytes(bool_array):
    """
    Convert an N x D boolean array into N strings of ceil(D/8) bytes each.
    """
    N, D = bool_array.shape
    # Pad each row to a multiple of 8
    padded_len = math.ceil(D / 8) * 8
    pad_width = padded_len - D
    if pad_width > 0:
        bool_array = np.pad(bool_array, ((0, 0), (0, pad_width)), constant_values=0)
    
    # Reshape to bytes and pack bits
    packed = np.packbits(bool_array, axis=1)
    # Convert each row to bytes
    return [row.tobytes() for row in packed]


def bytes_to_bools(byte_strings, D):
    """
    Convert back from N strings of ceil(D/8) bytes to an N x D boolean array.
    """
    bool_rows = []
    for b in byte_strings:
        row = np.unpackbits(np.frombuffer(b, dtype=np.uint8))[:D]
        bool_rows.append(row)
    return np.vstack(bool_rows).astype(bool)
    

def compute_color_LUT(circles, lines, target_image):
    """
    Regions are encoded as the binary string indicating which dividers they are on the positive side of.  For each divider,
    +1 indicates the point is on the positive side of the divider,
    
    positive side they are on.  (Circles come before lines in ordering, i.e. for circles, 
    div_index = circle_index, for lines, div_index = n_circles + line_index.)
        
    Pixel locations encode the same (ordered tuple of ints), and evaluate to the color of the region they're in.
    
    The color of each region is the average color of pixels in that region for the target image.
    Given the finite resolution of the target image, these regions might not cover the entire image area, so
    pixel locations in unencoded regions map to black (0,0,0) (FUTURE:  map to their closest matching code)
    
    Algorithm:  
        # Maintain a set of regions, for each a bit mask showing which x,y locations it encloses.
        code_to_region = {0: np.ones(h,w)}  # Initialize with the full image as one region
        last_region_index = 0
        
        For each div_ind, divider in enumerate(dividers = [circles + lines]):
            get its bitmask, see what coded region it crosses (has pixels on either side of the divider and in 
            the region).  For each such region region_x, split it into two new regions:
                - create regions last_region_index+1 and last_region_index+2 with appropriate bitmasks, add
                  to code_to_region
                - remove region_x from code_to_region.
                
    returns:  dict mapping code (tuple of ints) to color (3-vector)
    """
    h, w = target_image.shape[0], target_image.shape[1]
    x, y = make_input_grid((h, w), resolution=1.0, keep_aspect=True)
    x_lims = np.array((x.min(), x.max()))
    y_lims = np.array((y.min(), y.max()))
    codes_to_region = {0: np.ones((h, w), dtype=bool)}  # Initialize with the full image as one region
    last_region_index = 0
    div_index = 0
    n_circles = circles['centers'].shape[0]
    n_lines = lines['centers'].shape[0]
    
    def _find_regions_crossed(div_mask):
        """
        Find all regions in codes_to_region that are crossed by the given divider mask (i.e. have pixels on both sides of the divider).
        Returns a list of region indices and the two side masks for each split region.
        """
        crossed = []
        for code, region_mask in codes_to_region.items():
            side1 = np.logical_and(region_mask, div_mask)
            side2 = np.logical_and(region_mask, np.logical_not(div_mask))
            if np.any(side1) and np.any(side2):
                crossed.append((code, side1, side2))
        return crossed
    
    def split_regions(shape_mask):
        crossed_regions = _find_regions_crossed(shape_mask)
        for code, side1, side2 in crossed_regions:
            del codes_to_region[code]
            new_code1 = last_region_index + 1
            new_code2 = last_region_index + 2
            last_region_index += 2
            codes_to_region[new_code1] = side1
            codes_to_region[new_code2] = side2
            
        codes_to_region[code] = np.logical_or(side1, side2)
    
    for circle_index in range(circles['centers'].shape[0]):
        shape_mask = eval_circles(None, x, y, circles)[:, circle_index].reshape((h, w))
        split_regions(shape_mask)
            
    # now do lines
    for line_index in range(lines['centers'].shape[0]):
        shape_mask = eval_lines(None, x, y, lines)[:, line_index].reshape((h, w))
        split_regions(shape_mask)