import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
from util import make_input_grid


class TestImageMaker(object):
    """
    Methods with names starting with _synth_ create test images, their file prefixes
    are the methods names without the _synth_ part.
    Methods ending with _rand create different images each call.
    """
    def __init__(self, image_size_wh=(64, 48)): 
        self._wh = image_size_wh
        self._type_list = self._make_type_list()
        
    
    def _add_center_point(self, mask, center, center_size, x_grid, y_grid):
        """
        Mark a center_size x center_size square around the center as 0
        """
        
        
        x_size=( x_grid[1][0] - x_grid[0][0]) * center_size
        center_x = ((x_grid < center[0] + x_size / 2) & 
            (x_grid > center[0] - x_size / 2))

        y_size= (y_grid[0][1] - y_grid[0][0]) * center_size
        center_y = ((y_grid < center[1] + y_size / 2) & 
            (y_grid > center[1] - y_size / 2))
        mask[center_x & center_y] = 0
        
        return mask        
        
    def _make_circle_mask(self, center, radius, center_size):
        """
        Make a mask that is -1 inside the circle and 1 outside.
        it should be the same dimensions as the image.
        """
        
        center = center[::-1]  # xy to yx

        center[1] = -center[1]  # invert y axis
        y_grid, x_grid = make_input_grid((self._wh[1], self._wh[0]), resolution=1)
        # Compute the circle mask
        circle_mask = (y_grid - center[1]) ** 2 + (x_grid - center[0]) ** 2 - radius ** 2
        mask = np.where(circle_mask >= 0, 1, -1)
        if center_size > 0:
            mask = self._add_center_point(mask, center, center_size, x_grid, y_grid)
        return mask        
    
    def _make_line_mask(self, angle_rad, center, center_size=0):
        """
        Make a mask that is -1 on one side of the ine and 1 on the other.
        it should be the same dimensions as the image.
        """
        
        center = center[::-1]  # xy to yx

        center[1] = -center[1]  # invert y axis
        y_grid, x_grid = make_input_grid((self._wh[1], self._wh[0]), resolution=1)
        # Compute the line mask
        angle_rad -= np.pi / 2  # rotate by 90 degrees to get normal vector
        line_mask = (y_grid - center[1]) * np.cos(angle_rad) - (x_grid - center[0]) * np.sin(angle_rad)
        mask = np.where(line_mask >= 0, 1, -1)
        if center_size > 0:
            mask = self._add_center_point(mask, center, center_size, x_grid, y_grid)
        return mask
    
    def _mask_to_image_gray(self, mask):
        """
        Paint-by-numbers style conversion of a mask to an image.
        Count the number of distinct colors, compute n shades of gray.
        Sweep the mask from top to bottom, left to right, assigning shades
        in order to assign new numbers to shades.
        Then fill in the RGB image with the shades
        """
        
        unique_vals = np.unique(mask)
        n_colors = len(unique_vals)
        intensity = np.linspace(0, 255, n_colors, dtype=np.uint8)
        gray_shades = np.stack((intensity, intensity, intensity), axis=1)
        val_to_shade = {val: gray_shades[i] for i, val in enumerate(unique_vals)}
        img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=float)
        for val, shade in val_to_shade.items():
            img[mask == val] = shade
        return img.astype(np.uint8) 
    
    def _synth_line_rand(self):
        angle = np.random.uniform(0, 2 * np.pi)
        center = np.zeros(2)
        mask = self._make_line_mask(angle, center)
        return self._mask_to_image_gray(mask)
            
    def _synth_lines_2_rand(self):
        angle1 = np.random.uniform(0, 2 * np.pi)
        angle2 = angle1 + np.pi/3
        center = np.zeros(2)
        mask1 = self._make_line_mask(angle1, center)
        mask2 = self._make_line_mask(angle2, center)
        combined_mask = mask1*10 + mask2
        return self._mask_to_image_gray(combined_mask)
    
    def _synth_lines_3_rand(self):
        rad = .4
        angle_offset = np.random.uniform(0, 2 * np.pi)
        centers = np.array([(np.cos(t+angle_offset)*rad, 
                             np.sin(t+angle_offset)*rad)
                            for t in (0, 2*np.pi/3, 4*np.pi/3)])
        normals = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        angles = np.arctan2(normals[:,1], normals[:,0]) + np.pi/2
        
        mask1 = self._make_line_mask(angles[0], centers[0])
        mask2 = self._make_line_mask(angles[1], centers[1])
        mask3 = self._make_line_mask(angles[2], centers[2])
        combined_mask = mask1*10+  mask2*100 + mask3
        return self._mask_to_image_gray(combined_mask)
    
    def _synth_lines_4_rand(self):
        r=.4
        angles = np.radians([0, 90, 180, 270]) + np.pi/2
        angle_offset = np.random.uniform(0, 2 * np.pi)
        angles += angle_offset
        centers = np.array([(np.cos(t+angle_offset)*r, 
                             np.sin(t+angle_offset)*r)
                            for t in (0, np.pi/2, np.pi, 3*np.pi/2)])
        
        combined_mask = np.zeros((self._wh[1], self._wh[0]), dtype=int)
        for i, (angle, center) in enumerate(zip(angles, centers)):
            mask = self._make_line_mask(angle, center)
            combined_mask += mask * 10 **i
        return self._mask_to_image_gray(combined_mask)
    
    def _synth_circle_center(self):
        center = np.zeros(2)
        radius = 0.3
        mask = self._make_circle_mask(center, radius, center_size=0)
        return self._mask_to_image_gray(mask)
    
    def _help_synth_circles_rand(self, n=5):
        angle_offset = np.random.uniform(0, 2 * np.pi)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
        separation = 0.25
        radii = 0.5
        centers = np.array([(np.cos(t+angle_offset) * separation,
                             np.sin(t+angle_offset) * separation) for t in angles])
        masks = [self._make_circle_mask(center, radii, center_size=0)*10**i for i, center in enumerate(centers)]
        return self._mask_to_image_gray(np.sum(masks, axis=0))
    
    def _synth_circles_2_rand(self):
        return self._help_synth_circles_rand(n=2)
    def _synth_circles_3_rand(self):
        return self._help_synth_circles_rand(n=3)
    def _synth_circles_4_rand(self):
        return self._help_synth_circles_rand(n=4)

    def _func_prefix(self, func_name):
        func = getattr(self, func_name)
        return func.__name__[7:]  # strip off _synth_
        
    def _make_type_list(self):
        """
        Get a list of all class methods that make images (start with _synth_)
        """
        return [self._func_prefix(func) for func in dir(self) if func.startswith("_synth_")]

    def make_image(self, name):
        """
        Create an image of the specified type.
        """
        if name not in self._type_list:
            raise ValueError(f"Invalid type name: {name}, must be one of {self._type_list}")
        func = getattr(self, f"_synth_{name}")
        return func()
    
    def get_types(self):
        return self._type_list

def test_images():
    maker = TestImageMaker((640,480))
    test_types = maker.get_types()
    logging.info("Available test image types: %s" % (test_types,))
    n_types = len(test_types)
    n_cols = int(np.min((np.ceil(np.sqrt(n_types) *1.2), n_types)))
    n_rows = int(np.ceil(n_types / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    for i in range(n_types):
        img = maker.make_image(test_types[i])
        ax = axes[i]
        ax.imshow(img, cmap='gray' if img.ndim==2 else None)
        ax.set_title(f"{test_types[i]}", fontsize=14)
        ax.axis('off')
    # turn off unused axes
    for i in range(n_types, len(axes)):
        axes[i].axis('off')
    plt.suptitle("Test Images (size:  %i x %i)" % (maker._wh[0], maker._wh[1]), fontsize=16)
    plt.tight_layout()
    plt.show()
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    test_images()