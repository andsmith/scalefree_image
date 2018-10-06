# scalefree_image
Use TensorFlow to learn a scale-invariant approximation of an image, i.e defined by half-planes &amp; circles, etc.


#### Theory

Learn f(x,y) = (r,g,b) for all pixels of an image.

This is the "scale free" representation of the image. You can then create a new image from this representation
at any resolution,  or zoom out to see how the representation extrapolates.

To approximate the image, optimally place lines and/or circles in the plane and choose the color
for a given pixel based on what side of each line it is on (and/or whether it is inside or
outside each circle).  The color choosing step is done with ReLu units on top of the lines/circles.


#### Example

Input image:

![Input image](input/flower-roses-red-roses-bloom_small.jpg?raw=true "input")

Approximated with 100 lines and 500 color choosing ReLu units:

![Input image](output/output_130_redrose_lines_00000003.jpeg?raw=true "input")

Approximated with 100 circles and 500 ReLu units:

![Input image](output/output_131_redrose_circles_00000002.jpeg?raw=true "input")

Approximated with 100 lines and 100 circles and 500 ReLu units:

![Input image](output/output_132_redrose_circles_00000003.jpeg?raw=true "input")

Same as previous approximation, but higher resolution (`-x 5`)

![Input image](output/output_132_redrose_circles_00000005.jpeg?raw=true "input")


Line approximation, zoomed out, high res (`-b 3 -x 2.5`).

![Input image](output/output_130_redrose_lines_00000005_scaled.jpeg?raw=true "input")


#### Usage

`image_learn.py` saves the model after each set of epochs, to a .tf file, which includes the input image and all the training parameters.

You can resume training by loading a model with the `-m model.tf` option, and override the parameters contained therein with the other command-line options.

Run `python image_learn.py` to see all options.

NOTE:  Hit ESCAPE from the display window to shut down cleanly.  Program will exit after finishing the current epoch and saving the model.  Do NOT just close the window.  This is a VisPy bug where the close callback doesn't get called if the window is closed this way.

