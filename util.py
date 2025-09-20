import numpy as np



def make_input_grid( img_shape=None,resolution=1.0,margin=0.0):
    """ Make a grid of input coordinates in [-1,1]x[-1,1]
    img_shape: (height, width, channels)
    resolution: scaling factor for number of points (1.0 = one point per pixel)
    margin: extra margin around [-1,1]x[-1,1] (in input coordinates)
    """
    scale = margin + 1.0

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
    assert np.all(grid[:,0] >= -1.1) and np.all(grid[:,0] <= 1.1)
    assert np.all(grid[:,1] >= -1.1) and np.all(grid[:,1] <= 1.1)
    assert np.isclose(np.mean(grid[:,0]), 0.0, atol=0.01)
    assert np.isclose(np.mean(grid[:,1]), 0.0, atol=0.01)

if __name__ == "__main__":

    test_make_input_grid()
