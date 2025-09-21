import numpy as np

import cv2
import logging


def downscale_image(image, factor):
    if factor == 1.0:
        return image
    new_shape = (np.array(image.shape[:2]) / factor).astype(int)
    logging.info("Downsampling (factor = %.1f) image from %s to %s" % (factor, image.shape, new_shape))
    return cv2.resize(image, new_shape[::-1], interpolation=cv2.INTER_AREA)


def make_input_grid(img_shape=None, resolution=1.0, border=0.0):
    """ Make a grid of input coordinates in [-1,1]x[-1,1]
    img_shape: (height, width, channels)
    resolution: scaling factor for number of points (1.0 = one point per pixel)
    border: extra border around [-1,1]x[-1,1] (in input coordinates)
    """
    scale = border + 1.0

    h, w = img_shape[0]*resolution, img_shape[1]*resolution
    xs = (((np.arange(w, dtype=np.float32)+.5) / float(w) * 2.0 - 1.0) * scale)
    ys = (((np.arange(h, dtype=np.float32)+.5) / float(h) * 2.0 - 1.0) * scale)

    xv, yv = np.meshgrid(xs, ys)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_make_input_grid()
    logging.info("All tests passed.")
