### Local Background Correction

This background correction method works by averaging the pixels around each band and using this value as the unique background intensity for each band.

The `Pixel Sensitivity' value defines the radius of pixels to be considered around each band for the local background correction.  Generally, 5 seems to be a reasonable value, but choosing the optimal value can be dependent on your gel band spacing and overall gel conditions.

Otherwise, the correction system will run automatically - no other user input is required.