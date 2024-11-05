# iris-recognition

Replication of Iris detection and recognition paper.

## Design logic:

Implemented as described in _Personal Identification Based on Iris Texture Analysis_ by Ma et al.
using the CASIA Iris Image Database (version 1.0). Multiple iris images were first localized. Initially, we
tried thresholding subimages and then using Canny edge detection and Hough circles for both the pupil and
the iris. However, we switched to just using this method for pupil detection due to better results with a
more naive method: For the iris, we instead naively assumed it is concentrically outside
the pupil by estimating the iris radius to be 53 pixels longer than the pupil's and used this
to find its bounding circle. Then, eyelids were detected using parabola fitting. A mask containing
only the isolated iris was made and we ensured this was cropped and centered. Next, came iris
normalization. The localized iris images were used as input. A mapping was made to transform the
circular iris shape in polar coordinates into a 64x512 rectangle in cartesian coordinates. Then,
the normalized iris images were enhanced. We tried enhancing the image as described in the paper.
This involves first finding the mean of 16x16 blocks to estimate background illumination, performing
bicubic interpolation, subtracting this estimate from the normalized image, and finally enhancing the
image through histogram equalization on 32x32-sized blocks. However, we found that performance was 
better by simply performing histogram equalization once on the image as a whole. 


### FeatureExtraction.py
The FeatureExtraction.py file contains the following functions:

`spatial_filter`
This function is designed to initialize the defined spatial filter in Ma et al's work, which is created by multiplying the Gaussian envelope
with a sinusoidal modulating function to extract image features from a localized, normalized, then enhanced iris.
The spatial filter function takes as input the following parameters:

* `x`, which is the X-coordinate of the spatial filter grid.
* `y`, which is the Y-coordinate of the spatial filter grid.
* `sigma_x`, which is the standard deviation along the x-axis for the Gaussian envelope.
* `sigma_y`, which is the standard deviation along the y-axis for the Gaussian envelope.
* `f`, which is the frequency of the modulation function.

This function follows the below logic:
* The Gaussian envelope is initialized, per the formula provided by Ma et al. This is stored as the variable `G`
* The sinusoidal modulation function is initialized, per the formula provided by Ma et al. This is stored as the variable `Mi`
* `G` and `Mi` are multiplied together to obtain the complete spatial filter

`apply_spatial_filter`
This function is designed to apply the spatial filter initialized using the function above to an enhanced iris image. This function
also ensures that the spatial filter is centered at each pixel of the iris image before convolving it with the image.
The apply spatial filter function takes as an input the following parameters:

* `image`, which is the localized, normalized, then enhanced iris image as a numpy array
* `sigma_x`, which is the standard deviation along the x-axis for the Gaussian envelope
* `sigma_y`, which is the standard deviation along the y-axis for the Gaussian envelope
* `f`, which is the frequency of the modulating function.

This function follows the below logic:
* `np.fromfunction` is used to create an output image that is the correct size after the convolution. we do `x-rows//2` and `y-rows//2` in order to center
   the spatial filter at each pixel of the iris image.
* `cv2.filter2D` is used to convolve the centered spatial filter with the iris image.

`extract_features`
Using the frequency information that is the output of `apply_spatial_filter`, this function extracts the mean and absolute deviation
of each 8x8 patch of the filtered image. It rearranges these values into a 1D numpy vector, which will represent the iris
image.
The extract features function takes as an input the following parameters:

* `filtered_image`, which is the output of `apply_spatial_filter`. It is the frequency information of the iris image stored as a numpy array.

This function follows the below logic:
* Loops through every 8 pixels of the x-axis of the filtered image
* Loops through every 8 pixels of the y-axis of the filtered image
* Uses the indices at each iteration of the loop to section off an 8x8 portion of the filtered image. Calculates the mean and aboslute deviation
  for each patch.
* Flattens all of the stored means and absolute deviations into a 1D vector after both loops have finished running.

`feature_iris`
Uses the enhanced iris image (represented as a numpy array) and applies all four of the above functions in sequential order to the image.
This function mainly serves as a small pipeline to run all of the steps detailed above.
The feature iris function takes as an input the following parameters:

*`enhanced_iris`, which is the enhanced iris image (the output of the ImageEnhancement.py script) represented as a numpy array.

This function follows the below logic:
* Selects the Region Of Interest by using the top portion of the enhanced image
* Applies the spatial filter in two domains using the `apply_spatial_filter` function from above.
* Extracts the features from both filtered images using the `extract_features` function from above.
* Creates the final feature vector by concatenting the result of applying the filter in both domains.

## Limitation(s) of the current design:

Iris detection is not very smart due to the concentric circles assumption. 
Currently, enhancement is pretty basic and does not account for reflections or 
eyelashes.

## Improvements:

We could make improvements on locating the iris using the same thresholding/Canny edge/Hough
circle method with better parameters or improved image processing to help with the noise.
Image enhancement could be further improved by accounting for reflections and eyelashes.

## Peer evaluation:

IrisRecognition - Jake, Ellise, Nicole collaboration

IrisLocalization - Jake

IrisNormalization - Ellise

ImageEnhancement - Ellise

FeatureExtraction - Nicole

IrisMatching - Nicole

PerformanceEvaluation - Jake

Readme File - Jake, Ellise, Nicole collaboration

## Reference:

Ma et al., Personal Identification Based on Iris Texture Analysis, IEEE TRANSACTIONS ON
PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 25, NO. 12, DECEMBER 2003
