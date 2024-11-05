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

* `enhanced_iris`, which is the enhanced iris image (the output of the ImageEnhancement.py script) represented as a numpy array.

This function follows the below logic:
* Selects the Region Of Interest by using the top portion of the enhanced image
* Applies the spatial filter in two domains using the `apply_spatial_filter` function from above.
* Extracts the features from both filtered images using the `extract_features` function from above.
* Creates the final feature vector by concatenting the result of applying the filter in both domains.

### IrisMatching.py

The IrisMatching.py file contains the following functions:

`reduce_dimensionality`

This function uses the Fisher discriminant for dimension reduction of an iris feature vector. It then also determines the class centers
for each unique iris, which are also represented as 1D numpy vectors. 
The reduce dimensionality function takes as an input the following parameters:

* `features`, which is the iris feature vector obtained from running the `FeatureExtraction.py` file
* `labels`, which are the iris IDs obtained from extracting the first 3 digits of the file names of the iris images.
* `n_components`, which is the number of features we want to reduce the (1536, ) vector to. The maximum number of features we can 
   reduce to is the number of unique irises - 1.

This function follows the below logic:
* Initializes a `LinearDiscriminantAnalysis` model from scikits `disciminant_analysis` package with the specified number of components
* If no number of components is specified, the function does not perform dimension reduction. Rather, it just returns the original feature vectors.
* If a number of components is specified, `model.transform` is called to reduce the dimensionality of the original feature vectors to `n_components`
* Class centers for each iris are obtained by taking the mean of the feature vector where the labels match.

`compute_nearest_center`

This function classifies a feature vector by finding the nearest class center and returns the similarity measure for the highest probable class. This 
function also supports 3 different distance/similarity measures (L1, L2 and cosine). 
The compute nearest neighbor function takes as an input the following parameters:

* `reduced_feature`, which is the reduced feature vector obtained from the `reduced_dimensionality` function.
* `class_centers`, which are the class centers obtained from the `reduce_dimensionality` function.
* `distance_type`, which is a string specifying which type of distance/similarity to use when calculating the similarity score and best predicted class.

This function follows the below logic:
* Calculates the distance between each feature vector and each class center, and stores these values as a dictionary
* Converts the distances to a similarity score by doing 1 / distance (only applies to L1 and L2 distance)
* Chooses the index where the best similarity score is found, using softmax probabilities
* Uses the best index found to extract the predicted class from the labels, and the best similarity from the similarities

`match_iris`

This function matches a reduced feature vector to the best class by performing 7 different rotations of the feature vector (as specified by Ma et al.) The rotation that yields the closest match to one of the class centers is chosen, and the class associated with the closest class center is chosen as the class (and thus the iris) that the feature vector belongs to. The match iris function takes as input the following parameters:

* `feature`, which is the original feature vector of the iris
* `class_centers`, which are the class centers of the un-reduced feature vectors obtained from the `reduce_dimensionality` function.
* `reduced_class_centers`, which are the class centers of the reduced feature vectors obtained from the `reduce_dimensionality` function.
* `model`, which is the LDA model obtained from the `reduce_dimensionality` function.
* `rotations`, a list of angle rotations to perform on the feature vectors, obtained from the rotations specified in the work of Ma et al.
* `distance_type`, a string inidicating which type of distance / similarity to calculate.

This function follows the below logic:
* Initializes variables to keep track of the best similiarity score and best predicted class 
* Loops through each of the angles found in `rotations`. Uses `np.roll` to rotate the feature vectors based on the current angle.
* Reduces the dimensionality of the feature vector using `model.transform`, which is an LDA model for dimension reduction.
* For both the non-reduced and reduced feature vectors, calculates the predicted class and predicted similarity using `compute_nearest_center`
* For both the non-reduced and reduced feature vectors, tracks the best similarity, best class, and best feature vector based on if the angle rotation yields a closer match. The best metric of all rotations is returned.


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
