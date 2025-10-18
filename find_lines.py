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

class EdgeFinder(object):
    def __init__(self, image, n_levels=4):
        self.image = trim_image(image, n_pow_2=n_levels)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(self.gray, 50, 150)
        self.n_levels = n_levels
        self.build_pyramid()
        self.make_edge_masks()
        
        
    def make_edge_masks(self):
        self.edge_masks = []
        block_template = np.ones((2,2),np.uint8)
        for l, edges in enumerate(self.edge_pyr):
            edge_mask = (edges > 128).astype(np.uint8) # count, so add 1
            while edge_mask.shape[0] < self.edge_pyr[0].shape[0] or edge_mask.shape[1] < self.edge_pyr[0].shape[1]:
                edge_mask = np.kron(edge_mask, block_template)
            self.edge_masks.append(edge_mask)
            
        print("Made %i edge masks" % len(self.edge_masks))
        print("Shapes: ", [lvl.shape for lvl in self.edge_masks])
        
    def build_pyramid(self):
        self.img_pyr = []
        self.edge_pyr = []
        gray = self.gray.copy()
        for _ in range(self.n_levels):
            self.img_pyr.append(gray)
            edges = cv2.Canny(gray, 50, 150)
            self.edge_pyr.append(edges)
            gray = cv2.pyrDown(gray)
        print("Made pyramid with %i levels" % len(self.img_pyr))
        print("Sizes: ", [lvl.shape for lvl in self.img_pyr])
        
        
    def optimize(self):
        """
        Find the optimal x,y offsets for each edge mask to align lines.
        Images are best aligned when the sum over all masks+alignments is maximized.
        edge mask on level n_levels can't be moved.
        Edge mask on level n_level-1 can be moved +/- 2 pixels in x and y
        Edge mask on level n_level-2 can be moved +/- 4 pixels in x and y
        etc.
        """
        level_offsets = []
        





image = (cv2.imread(sys.argv[1])[:,:,::-1] )
test = EdgeFinder(image, n_levels=6)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_pyr = []
edge_pyr = []
while gray.shape[0] > 16 and gray.shape[1] > 16:
    image_pyr.append(gray)
    edge_pyr.append(cv2.Canny(gray, 50, 150))
    gray = cv2.pyrDown(gray)
    
print("Made pyramid with %i levels" % len(image_pyr))
print("Sizes: ", [lvl.shape for lvl in image_pyr])


n_levels = 4

block_template = np.ones((2,2),np.uint8)

    
    

line_mask = np.zeros_like(edge_pyr[0], dtype=np.uint8)
edge_masks_resized = []
for l, edges in enumerate(edge_pyr[:n_levels]):
    print("Adding edges from level %i with shape %s" % (l, str(edges.shape)))
    # resize to original level, add to line_mask
    edge_mask = (edges > 128).astype(np.uint8) # count, so add 1
    while edge_mask.shape[0] < line_mask.shape[0] or edge_mask.shape[1] < line_mask.shape[1]:
        edge_mask = np.kron(edge_mask, block_template)
    try:
        line_mask += edge_mask
        edge_masks_resized.append(edge_mask)
    except Exception:
        print("Pyramid wrong shape at level, %i, stopping." % l)
        break
    
    
    
img_extent = (0, line_mask.shape[1], line_mask.shape[0], 0)
# Plot used edge levels in fig 1
fig, ax = plt.subplots(nrows=2, ncols=n_levels, sharex=True, sharey=True)
for e_l, edges in enumerate(edge_pyr[:n_levels]):
    ax[0,e_l].imshow(edges, cmap='viridis', extent=img_extent)
    ax[0,e_l].set_title(f"Edges level {e_l}")
 
    if e_l < len(edge_masks_resized):
        ax[1,e_l].imshow(edge_masks_resized[e_l], cmap='viridis', extent=img_extent)
        ax[1,e_l].set_title(f"Resized edges level {e_l}")
 
    

# Plot combined line mask in fig 2
fig, lax = plt.subplots()
lax.imshow(line_mask, cmap='viridis')
lax.set_title("Combined line mask")
colorbar = plt.colorbar(lax.images[0], ax=lax, orientation='vertical')
    




# import ipdb; ipdb.set_trace()

# # Apply Probabilistic Hough Line Transform
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)


# fig, ax=plt.subplots(2,2)
# ax=ax.flatten()


# # Draw the detected lines on the original image
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         plt.plot([x1, x2], [y1, y2], 'r-')

# plt.title('Detected Lines')
# plt.axis('off')
# plt.axis('equal')
plt.show()
