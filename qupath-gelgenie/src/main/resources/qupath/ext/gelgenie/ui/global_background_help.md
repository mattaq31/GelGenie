### Global Background Correction

This background correction method works by subtracting a constant `global' background value from all pixels in your gel bands.  The global value is obtained by averaging the intensity of pixels in a user-defined background patch on the gel image.  To use this method:

1. Create an annotation covering a background region on your gel. This can be any shape, but creating a rectangle is the simplest method.

2.  With the annotation selected, click on the 'Set background region' button to register the annotation as your global background (the annotation will turn blue).

3. Activate the method's checkbox.

That's it - the result will be automatically computed and added to your results table! 