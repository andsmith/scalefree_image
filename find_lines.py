"""
find_lines.py

Try to find lines dividers for a given image by looking for lines directly.

1.  


"""

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def trim_image(image, n_pow_2 = 4):
    """ Trim image so that both dimensions are multiples of 2**n_pow_2
    """

    h, w = image.shape[:2]
    new_h = (h // (2**n_pow_2)) * (2**n_pow_2)
    new_w = (w // (2**n_pow_2)) * (2**n_pow_2)
    print("Trimming image by (x,y)=%s, sizes for %i levels will be:  \n%s "% ( (w - new_w, h - new_h),n_pow_2, np.array((new_h, new_w))/2**np.arange(n_pow_2).reshape(-1,1)))
    trimmed_image = image[:new_h, :new_w]
    return trimmed_image

def pyr_down(image):
    return( (image[::2, ::2].astype(int)+image[1::2, ::2]+image[::2, 1::2]+ image[1::2, 1::2]) //4).astype(np.uint8)


class EdgeFinder(object):
    def __init__(self, image, n_levels=4):
        self.image = trim_image(image, n_pow_2=n_levels)

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(self.gray, 50, 150)
        self.n_levels = n_levels
        self.build_pyramid()
        self.make_edge_masks()
        self.combined_mask = np.sum(np.array(self.edge_masks), axis=0).astype(np.uint8) 
        self._bin_mask = (self.combined_mask >=np.max(self.combined_mask)).astype(np.uint8)

        self.lines = cv2.HoughLinesP(self._bin_mask, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=30)
        print("Found %i lines" % (0 if self.lines is None else len(self.lines)))
        
        

        
    def plot(self):
        """
        Open 2 figures.  In 1 show the combined edge mask and a colorbar.
        In the other show a 2 column plot with all the edge masks.
        """
        fig, ax = plt.subplots(ncols=3)
        ax[0].imshow(self.combined_mask, cmap='viridis')
        plt.colorbar(ax[0].images[0], ax=ax[0], orientation='vertical')
        ax[1].imshow(self.image)
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            ax[1].plot([x1, x2], [y1, y2], 'r-')    
            
        ax[2].imshow(self._bin_mask, )
        ax[2].set_title("Binary Mask Used for Hough")
        
        
        plt.title("Combined Edge Mask")
        n_cols=2
        n_rows = int(np.ceil(self.n_levels / n_cols))

        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True)
        ax = ax.flatten()
        for e_l, edges in enumerate(self.edge_masks):
            ax[e_l ].imshow(edges, cmap='viridis')
            ax[e_l ].set_title(f"Edges level {e_l}")
        plt.suptitle("Edge Masks at Different Levels")
        
    def get_edges(self, thresh):
        return self.edges > thresh

    def make_edge_masks(self):
        self.edge_mask_pyramid = []
        self.edge_masks = []  # resized versions
        block_template = np.ones((2,2),np.uint8)
        for l, edges in enumerate(self.edge_pyr):
            edge_mask = (edges > 128).astype(np.uint8) # count, so add 1
            self.edge_mask_pyramid.append(edge_mask)
            while edge_mask.shape[0] < self.edge_pyr[0].shape[0] or edge_mask.shape[1] < self.edge_pyr[0].shape[1]:
                edge_mask = np.kron(edge_mask, block_template)
            self.edge_masks.append(edge_mask)

        print("Made %i edge masks" % len(self.edge_mask_pyramid))
        print("Shapes: ", [lvl.shape for lvl in self.edge_mask_pyramid])

    def build_pyramid(self):
        self.img_pyr = []
        self.edge_pyr = []
        gray = self.gray.copy()
        for _ in range(self.n_levels):
            self.img_pyr.append(gray)
            edges = cv2.Canny(gray, 50, 150)
            self.edge_pyr.append(edges)
            gray = pyr_down(gray)
        print("Made pyramid with %i levels" % len(self.img_pyr))
        print("Sizes: ", [lvl.shape for lvl in self.img_pyr])
        
if __name__ == "__main__":

    image = trim_image(cv2.imread(sys.argv[1])[:,:,::-1],2)
    test = EdgeFinder(image, n_levels=1)
    test.plot()
    plt.show()
