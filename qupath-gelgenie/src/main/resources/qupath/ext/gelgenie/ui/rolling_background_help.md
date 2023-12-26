### Rolling-Ball Background Correction

This background correction method works by calculating the background for the entire image using ImageJ's implementation of the 'rolling ball' algorithm.  This background is then subtracted individually for each pixel in the band of interest.  More details can be found here: https://imagej.net/ij/docs/menus/process.html#background.

The rolling ball radius can be user-defined but otherwise the method will run automatically (no additional user input is required).