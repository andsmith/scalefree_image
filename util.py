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

def add_text(image, text_lines, bbox, line_spacing = 1.5, max_font_scale=3.0, min_font_scale=0.1, 
             font_face=cv2.FONT_HERSHEY_SIMPLEX, font_thickness=1, color=(128,128,128), justify='center'):
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
    max_lines = 0
    width = x1 - x0
    max_text_w = width *2.0
    while len(text_lines) > max_lines or max_text_w > width and font_scale > min_font_scale:
        font_scale -= 0.01
        widths, heights = [],[]
        for line in text_lines:
            ((text_w, text_h), _) = cv2.getTextSize(line, font_face, font_scale, font_thickness)
            widths.append(text_w)
            heights.append(text_h)
        max_text_w = max(widths)
        text_h = max(heights)

        line_height = int(text_h * line_spacing)
        max_lines = (y1 - y0) // line_height    

    font_scale = np.clip(font_scale, min_font_scale, max_font_scale)


    total_text_height = len(text_lines) * line_height
    y_start = y0 + (y1 - y0 - total_text_height) // 2 + text_h

    for i, line in enumerate(text_lines):
        y = y_start + i * line_height
        ((text_w, text_h), _) = cv2.getTextSize(line, font_face, font_scale, font_thickness)
        if justify == 'left':
            x = x0 + 5  # small left margin
        elif justify == 'right':
            x = x1 - text_w - 5  # small right margin
        elif justify == 'center':
            x = x0 + (x1 - x0 - text_w) // 2  # center text horizontally
        else:
            raise ValueError("Unknown justify option: %s" % justify)
        cv2.putText(image, line, (x, y), font_face, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    #draw_bbox(image, bbox, color=(0, 255, 0), thickness=1)


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=1):
    """ Draw a bounding box on an image
    image: HxWx3 BGR image
    bbox: {'x': (x-min, x-max), 'y': (y-min, y-max)} bounding box in image coordinates
    color: BGR color tuple
    thickness: line thickness
    """
    (x0, x1), (y0, y1) = bbox['x'], bbox['y']
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
        

def captioned_frame(img, caption, caption_height_px=30, caption_pad_xy=(10, 5), txt_color=(255,255,255), bkg_color=(0,0,0), **kwargs):
    caption_h = caption_height_px
    pad_x, pad_y = caption_pad_xy
    img_cap = np.zeros((img.shape[0]+caption_h, img.shape[1], 3), dtype=np.uint8)
    img_cap[:img.shape[0], :, :] = img
    img_cap[img.shape[0]:, :, :] = bkg_color
    img_cap[:img.shape[0], :img.shape[1], :] = img
    bbox = {'x': (pad_x, img.shape[1]-pad_x),
            'y': (img.shape[0]+int(pad_y*1.5), img_cap.shape[0]-(pad_y//2))}
    add_text(img_cap, caption, bbox, font_face=cv2.FONT_HERSHEY_SIMPLEX, justify='center', color=txt_color, **kwargs)
    return img_cap

def test_add_text():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    bbox = {'x': (50, 350), 'y': (50, 150)}
    text_lines = ["This is a test", "of the add_text function.", "It should center text", "within the bounding box."]
    add_text(img, text_lines, bbox, line_spacing=1.5)
    cv2.rectangle(img, (bbox['x'][0], bbox['y'][0]), (bbox['x'][1], bbox['y'][1]), (0, 255, 0), 1)
    cv2.imshow("Test Add Text", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_captioned_frame():
    
    test_frame = cv2.imread('movies\\washington_linear_8d_10h_cycle-00000010.png')
    caption = ['This is a test caption', 'Second line of caption']
    frame = captioned_frame(test_frame, caption, caption_height_px=50, line_spacing=2.0,caption_pad_xy=(10, 5), txt_color=(255,255,255), bkg_color=(50,50,50))

    cv2.imshow("Test Captioned Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_captioned_frame()
    logging.info("All tests passed.")
