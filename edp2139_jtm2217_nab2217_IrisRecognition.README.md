# iris-recognition

Replication of Iris detection and recognition paper.

## Design logic:

Implemented as described in _Personal Identification Based on Iris Texture Analysis_ by Ma et al. using the CASIA Iris Image Database (version 1.0).

# IrisReconition.py

**<span style="color: red;">Run Instructions:</span>**

- Insert the CASIA database as a folder named "database" into the `iris-recognition` directory.
- In your command line (ensuring that you are in the `iris-recognition` directory), run the command `python IrisRecognition.py`.
- After these two steps are performed, the outputs will be tables printed in the console and figures that will open automatically.

The IrisRecognition.py file contains the following functions:

`main()`

- This function does not accept any arguments, but runs the entire project.

This function follows the below logic:

- Processes all of the images, and stores them into training and test sets. In order to obtain the class labels, the images IDs are extracted by taking the first 3 digits of each image file name.
- In the following order, `IrisLocalization.py`, `ImageNormalization.py`, `ImageEnhancement.py`, and `FeatureExtraction.py` are all run. This essentially serves as a processing pipeline where the irises are first localized, normalized, enhanced, then converted to a 1D feature vector. More detailed descriptions for each of these files are contained below.
- These four files are run on both the training and the test sets.
- If these four files have already been run once, there is no need to do it again. The function will simply load in pre-saved feature vectors for both the train and the test sets, obtained from running the pre-processing pipeline the first time.
- Next, an LDA model is fit to the training set (both the reduced and non-reduced feature vectors). We use these models to perform dimensionality reduction on the training set, as well as the test set so that there is no information leakage.
- For each distance type (L1, L2, and cosine) the reduced feature vectors in the training set are assigned to the iris with the closes match based on similarity. This is done by running the functions contained in `IrisMatching.py`, which are described in further detail below.
- For each number of dimensions we wanted to try reducing to [30, 60, 80, 100, 107], as well as for each distance type (L1, L2, and cosine) the reduced feature vectors in the training set are assigned to the iris with the closes match based on similarity. This is done by running the functions contained in `IrisMatching.py`, which are described in further detail below. For the test set, similarity scores, class labels, and Correct Recognition Rates are stored in lists for plotting later.
- Performance evaluation for Correct Recognition Rate is performed by running functions from `PerformanceEvaluation.py` on the stored lists of metrics. A more detailed description of the functions contained in this file are detailed below.
- Performance evaluation for False Match Rate and False Non Match Rate is performed by running functions from `PerformanceEvaluation.py` on the stored lists of metrics. A more detailed description of the functions contained in this file are detailed below.

# IrisLocalization.py

## Logic Overview:

This script is designed to locate and return the segmented irises. It is designed to accurately detect and outline the iris region in a grayscale eye image, with a specific focus on locating the iris in relation to the pupil. The function first estimates a region around a determined pupil center by estimating the pupil center, projecting in the vertical and horizontal direction. It extracts a sub-image around this area and applies histogram equalization to enhance the visibility of iris features. Next, it performs binary thresholding to highlight the iris and removes noise with morphological operations, such as dilation and erosion, ensuring that any surrounding eyelashes or reflections do not interfere with detection. Using canny edge detection, the script identifies the circular boundaries of the iris by locating the contour that best matches a circular shape around the pupil center using Hough Circle detection.

## Logic and Parameters by Function:

### `load_and_preprocess_image`:

- **Logic**: Loads image, converts to grayscale, and applies Gaussian blur to reduce noise.
- **Parameters**:
  - `image_path`: Path to input image file.
  - `gray_image`: Grayscale version of input image.
- **Returns**: Tuple of `(original_image, preprocessed_image)`.

---

### `detect_pupil`:

- **Logic**: Uses projection and binary thresholding to estimate pupil center, uses Canny Edge detection on the thresholded image, and applies Hough Circle Transform.
- **Parameters**:
  - `param1`: Edge detection threshold (derived from contrast).
  - `param2`: Circle detection threshold.
  - `circles`: Detected circles from Hough transform.
  - `edges`: Edge-detected image.
  - `pad_size`: Amount of padding (50 pixels) added to handle boundary cases.
- **Returns**: Tuple of `(pupil_circle, center)`.

---

### `detect_iris`:

- **Logic**: Detects iris boundary using edge detection and Hough Transform.
- **Parameters**:
  - `attempt2`: Thresholded image for iris detection.
  - `edges`: Edge-detected image.
  - `lower_threshold, upper_threshold`: Thresholds based on image statistics.
  - `contrast`: Standard deviation of edges.
- **Returns**: Tuple of `(iris_circle, final_center)`.

---

### `locate_iris`:

- **Logic**: Main function coordinating pupil and iris detection, creates masks for segmentation.
- **Parameters**:
  - `mask_between`: Mask showing area between iris and pupil.
  - `segmented`: Final segmented iris image.
  - `box_size`: Size of bounding box (2.5 times iris radius).
  - `centered_segmented`: Cropped and centered final image.

---

### `detect_eyelids`:

- **Logic**: Fits parabolic curves to eyelid edges and creates masks to remove eyelid regions.
- **Parameters**:
  - `upper_edge_points, lower_edge_points`: Points for eyelid edges.
  - `parabola_upper, parabola_lower`: Coefficients of fitted curves.
  - `mask`: Binary mask for removing eyelid regions.
- **Returns**: Image with eyelids masked out.

### `find_thresholded_subimage(image, center, lower_bound, upper_bound, size=120, is_pupil=True, repeat=False)`

- **Purpose**: Extracts a thresholded sub-image around a specified center point, applying various preprocessing and thresholding techniques to enhance and segment the region of interest, which can be either the pupil or the iris.
- **Parameters**:

  - `image` (`numpy.ndarray`): The input grayscale image from which the sub-image will be extracted.
  - `center` (`tuple`): Coordinates `(x, y)` representing the estimated center around which the sub-image will be centered.
  - `lower_bound` (`int`): The lower bound for binary thresholding.
  - `upper_bound` (`int`): The upper bound for binary thresholding.
  - `size` (`int`, optional): The size of the square sub-image. Defaults to 120 pixels.
  - `is_pupil` (`bool`, optional): Flag indicating if the sub-image is for the pupil (`True`) or the iris (`False`), adjusting preprocessing methods accordingly. Defaults to `True`.
  - `repeat` (`bool`, optional): Flag for whether this is a second attempt at thresholding. If `True`, applies additional morphological operations to refine the binary image. Defaults to `False`.

- **Returns**:

  - `thresh` (`numpy.ndarray`): The processed binary thresholded sub-image.
  - `(cx, cy)` (`tuple`): Centroid coordinates of the thresholded area within the sub-image.
  - `(x_start, y_start)` (`tuple`): Top-left coordinates of the sub-image within the original image.

- **Logic**:
  1. **Extract Sub-image**:
     - Based on the center coordinates and specified `size`, a square sub-image is cropped from the original image.
  2. **Preprocessing and Thresholding**:
     - If `is_pupil` is `True`:
       - **Blur and CLAHE**: Applies Gaussian blur to reduce noise (e.g., from eyelashes) and CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast.
       - **Gamma Correction**: If mean brightness is high, adjusts gamma and darkens the image to better highlight pupil features.
       - **Binary Thresholding**: Converts the sub-image to binary using the specified `lower_bound` and `upper_bound`.
       - **Morphological Operations**: Uses dilation and erosion to eliminate noise (like eyelashes) and applies morphological closing to smooth boundaries.
     - If `is_pupil` is `False`:
       - **Equalization and CLAHE**: Uses histogram equalization and CLAHE to enhance contrast before thresholding.
       - **Binary Thresholding**: Directly thresholds the image with specified bounds.
  3. **Calculate Centroid**:
     - Calculates the centroid of the thresholded area using image moments to determine the center of the thresholded region.

---

## Supporting Functions:

### `adjust_gamma`: Adjusts image brightness.

- **Parameters**:
  - `gamma`: Brightness adjustment factor (default 1.5).

### `darken_blacks`: Darkens specific regions.

- **Parameters**:
  - `threshold_value`: Intensity threshold (default 50).
  - `darkening_factor`: Factor for darkening pixels (default 0.5).

### `pad_image`: Adds padding to image.

- **Parameters**:
  - `pad_size`: Number of pixels to add on each side.

---

# IrisNormalization.py

The IrisNormalization.py file contains the following functions:

`ensure_gray`
This function ensures the image was loaded correctly and in grayscale. If so, it returns the image. Otherwise, it
converts the image to grayscale and returns that. If the image is not loaded correctly, an error message is
displayed and the program is exited. This is done because grayscale images are necessary for multiple cv2 utilities.
It takes as input the following parameters:

- `img`, which is an nparray containing the image in question

`calc_boundary_point`
This function calculates the boundary points (x, y) at a given angle in a circle. It then returns these boundary points.
It takes as input the following parameters:

- `theta`, which is the given angle in radians as an np.float64
- `circle`, which is the circle to find boundary points from. It is a list containing the x_coord, y_coord, & radius of the circle.

`normalize_iris`
This function maps the localized iris image from Cartesian coordinates to polar coordinates. It calls the IrisLocalization file to
get both the pupil and iris circle boundaries. It initializes an empty nparray of size 64x512 (as suggested in the paper)
to hold the normalized iris image. Then, it divides the 360-degree polar coordinate space into 512 angle segments and iterates over
each angle around the circle. At each theta angle, the pupil and iris boundary points are calculated using `calc_boundary_point`. Then,
it iterates 64 times (corresponding to the normalized cartesian map size) each time interpolating points between the two boundaries.
This is to develop a representation of the polar space iris in the cartesian space. Lastly, it is mapped to the original image and a
normalized iris image is returned. It takes as input the following parameters:

- `mask`, which is an nparray of a localized iris mask image

# ImageEnhancement.py

The ImageEnhancement.py file contains the following functions:

`enhance_iris_basic`
This function first ensures the given image is in grayscale, and then calls cv2.equalizeHist histogram equalization to enhance the image.
It then returns the enhanced image. It is a simple approach, but seems to work very well. It takes as input the following parameters:

- `norm_iris`, which should be an nparray of a normalized iris image

`enhance_iris`
This function enhances the image as described in the paper. This involves first finding the mean of 16x16 blocks to estimate background illumination, performing bicubic interpolation, subtracting this estimate from the normalized image, and finally enhancing the
image through histogram equalization on 32x32-sized blocks. However, we found that performance was
better by simply performing histogram equalization once on the image as a whole as in `enhance_iris_basic`. This function did not end up being used. Regardless, it takes as input the following parameters:

# FeatureExtraction.py

The FeatureExtraction.py file contains the following functions:

`spatial_filter`

This function is designed to initialize the defined spatial filter in Ma et al's work, which is created by multiplying the Gaussian envelope
with a sinusoidal modulating function to extract image features from a localized, normalized, then enhanced iris.
The spatial filter function takes as input the following parameters:

- `x`, which is the X-coordinate of the spatial filter grid.
- `y`, which is the Y-coordinate of the spatial filter grid.
- `sigma_x`, which is the standard deviation along the x-axis for the Gaussian envelope.
- `sigma_y`, which is the standard deviation along the y-axis for the Gaussian envelope.
- `f`, which is the frequency of the modulation function.

This function follows the below logic:

- The Gaussian envelope is initialized, per the formula provided by Ma et al. This is stored as the variable `G`
- The sinusoidal modulation function is initialized, per the formula provided by Ma et al. This is stored as the variable `Mi`
- `G` and `Mi` are multiplied together to obtain the complete spatial filter

`apply_spatial_filter`

This function is designed to apply the spatial filter initialized using the function above to an enhanced iris image. This function
also ensures that the spatial filter is centered at each pixel of the iris image before convolving it with the image.
The apply spatial filter function takes as an input the following parameters:

- `image`, which is the localized, normalized, then enhanced iris image as a numpy array
- `sigma_x`, which is the standard deviation along the x-axis for the Gaussian envelope
- `sigma_y`, which is the standard deviation along the y-axis for the Gaussian envelope
- `f`, which is the frequency of the modulating function.

This function follows the below logic:

- `np.fromfunction` is used to create an output image that is the correct size after the convolution. we do `x-rows//2` and `y-rows//2` in order to center
  the spatial filter at each pixel of the iris image.
- `cv2.filter2D` is used to convolve the centered spatial filter with the iris image.

`extract_features`

Using the frequency information that is the output of `apply_spatial_filter`, this function extracts the mean and absolute deviation
of each 8x8 patch of the filtered image. It rearranges these values into a 1D numpy vector, which will represent the iris
image.
The extract features function takes as an input the following parameters:

- `filtered_image`, which is the output of `apply_spatial_filter`. It is the frequency information of the iris image stored as a numpy array.

This function follows the below logic:

- Loops through every 8 pixels of the x-axis of the filtered image
- Loops through every 8 pixels of the y-axis of the filtered image
- Uses the indices at each iteration of the loop to section off an 8x8 portion of the filtered image. Calculates the mean and aboslute deviation
  for each patch.
- Flattens all of the stored means and absolute deviations into a 1D vector after both loops have finished running.

`feature_iris`

Uses the enhanced iris image (represented as a numpy array) and applies all four of the above functions in sequential order to the image.
This function mainly serves as a small pipeline to run all of the steps detailed above.
The feature iris function takes as an input the following parameters:

\* `enhanced_iris`, which is the enhanced iris image (the output of the ImageEnhancement.py script) represented as a numpy array.

This function follows the below logic:

- Selects the Region Of Interest by using the top portion of the enhanced image
- Applies the spatial filter in two domains using the `apply_spatial_filter` function from above.
- Extracts the features from both filtered images using the `extract_features` function from above.
- Creates the final feature vector by concatenting the result of applying the filter in both domains.

# IrisMatching.py

The IrisMatching.py file contains the following functions:

`reduce_dimensionality`

This function uses the Fisher discriminant for dimension reduction of an iris feature vector. It then also determines the class centers
for each unique iris, which are also represented as 1D numpy vectors.
The reduce dimensionality function takes as an input the following parameters:

- `features`, which is the iris feature vector obtained from running the `FeatureExtraction.py` file
- `labels`, which are the iris IDs obtained from extracting the first 3 digits of the file names of the iris images.
- `n_components`, which is the number of features we want to reduce the (1536, ) vector to. The maximum number of features we can
  reduce to is the number of unique irises - 1.

This function follows the below logic:

- Initializes a `LinearDiscriminantAnalysis` model from scikits `disciminant_analysis` package with the specified number of components
- If no number of components is specified, the function does not perform dimension reduction. Rather, it just returns the original feature vectors.
- If a number of components is specified, `model.transform` is called to reduce the dimensionality of the original feature vectors to `n_components`
- Class centers for each iris are obtained by taking the mean of the feature vector where the labels match.

`compute_nearest_center`

This function classifies a feature vector by finding the nearest class center and returns the similarity measure for the highest probable class. This
function also supports 3 different distance/similarity measures (L1, L2 and cosine).
The compute nearest neighbor function takes as an input the following parameters:

- `reduced_feature`, which is the reduced feature vector obtained from the `reduced_dimensionality` function.
- `class_centers`, which are the class centers obtained from the `reduce_dimensionality` function.
- `distance_type`, which is a string specifying which type of distance/similarity to use when calculating the similarity score and best predicted class.

This function follows the below logic:

- Calculates the distance between each feature vector and each class center, and stores these values as a dictionary
- Converts the distances to a similarity score by doing 1 / distance (only applies to L1 and L2 distance)
- Chooses the index where the best similarity score is found, using softmax probabilities
- Uses the best index found to extract the predicted class from the labels, and the best similarity from the similarities

`match_iris`

This function matches a reduced feature vector to the best class by performing 7 different rotations of the feature vector (as specified by Ma et al.) The rotation that yields the closest match to one of the class centers is chosen, and the class associated with the closest class center is chosen as the class (and thus the iris) that the feature vector belongs to. The match iris function takes as input the following parameters:

- `feature`, which is the original feature vector of the iris
- `class_centers`, which are the class centers of the un-reduced feature vectors obtained from the `reduce_dimensionality` function.
- `reduced_class_centers`, which are the class centers of the reduced feature vectors obtained from the `reduce_dimensionality` function.
- `model`, which is the LDA model obtained from the `reduce_dimensionality` function.
- `rotations`, a list of angle rotations to perform on the feature vectors, obtained from the rotations specified in the work of Ma et al.
- `distance_type`, a string inidicating which type of distance / similarity to calculate.

This function follows the below logic:

- Initializes variables to keep track of the best similiarity score and best predicted class
- Loops through each of the angles found in `rotations`. Uses `np.roll` to rotate the feature vectors based on the current angle.
- Reduces the dimensionality of the feature vector using `model.transform`, which is an LDA model for dimension reduction.
- For both the non-reduced and reduced feature vectors, calculates the predicted class and predicted similarity using `compute_nearest_center`
- For both the non-reduced and reduced feature vectors, tracks the best similarity, best class, and best feature vector based on if the angle rotation yields a closer match. The best metric of all rotations is returned.

# PerformanceEvaluation.py

## Logic Overview:

This script is designed to evaluate the performance of an iris recognition system using various metrics
It includes functions for calculating recognition rates, plotting performance curves, and creating comparison tables
The main metrics used are Correct Recognition Rate (CRR), False Match Rate (FMR), and False Non-Match Rate (FNMR)

## Logic and Parameters by Function:

### `CRR(preds, labels):`

- **Logic**: Calculates percentage of correct predictions by comparing predicted and actual labels.
- **Parameters**:
  - `preds`: Array of predicted class labels.
  - `labels`: Array of true/actual class labels.
- **Returns**: Accuracy percentage.

---

### `plot_CRR_curves(results):`

- **Logic**: Creates 3 subplots comparing CRR across different similarity measures (L1, L2, Cosine).
- **Parameters**:
  - `results`: DataFrame containing performance metrics.
  - `fig, axs`: Figure and subplot axes for visualization.
  - `x`: Array of dimension numbers for x-axis values.
- **Columns used**:
  - `crr_l1_reduced`: CRR values for L1 similarity.
  - `crr_l2_reduced`: CRR values for L2 similarity.
  - `crr_cosine_reduced`: CRR values for cosine similarity.

---

### `false_rate(similarity, labels, threshold, preds):`

- **Logic**: Calculates both FMR (False Match Rate) and FNMR (False Non-Match Rate) based on similarity scores.
- **Parameters**:
  - `similarity`: Array of similarity scores between pairs.
  - `labels`: True class labels.
  - `threshold`: Decision threshold for determining matches.
  - `preds`: Predicted class labels.
- **Returns**: Tuple of `(false_match_rate, false_non_match_rate)`.

---

### `plot_ROC(fmr, fnmr):`

- **Logic**: Plots Receiver Operating Characteristic curve showing relationship between FMR and FNMR.
- **Parameters**:
  - `fmr`: Array of False Match Rate values.
  - `fnmr`: Array of False Non-Match Rate values.

---

### `print_CRR_tables(CRR_RESULTS)`

- **Purpose**: This function displays Cross Recognition Rate (CRR) tables for different dimensions. Each table includes CRR values based on different similarity measures and for both normal and reduced data.

- **Parameters**:

  - `CRR_RESULTS` (`pd.DataFrame`): A DataFrame containing CRR values. The DataFrame should have an index representing different dimensions, and columns for CRR values with similarity measures `L1`, `L2`, and `Cosine`, in both normal and reduced forms (e.g., `crr_l1_normal`, `crr_l1_reduced`).

- **Logic**:
  1. **Iterate Over Dimensions**: Loops through each unique dimension in the `CRR_RESULTS` index.
  2. **Filter Data for Dimension**: Extracts data for the current dimension, creating a table containing CRR values.
  3. **Format Table**: Constructs a new DataFrame with similarity measures as rows (`L1`, `L2`, and `Cosine`) and columns for `CRR Normal (%)` and `CRR Reduced (%)`.
  4. **Display**: Prints the table along with a title indicating the dimension and a separator for readability.

---

### `create_fmr_fnmr_table(thresholds, fmr_list, fnmr_list)`

- **Purpose**: This function creates and displays a table of False Match Rate (FMR) and False Non-Match Rate (FNMR) for a range of threshold values.

- **Parameters**:

  - `thresholds` (`list`): A list of threshold values used to compute FMR and FNMR.
  - `fmr_list` (`list`): A list of FMR values corresponding to each threshold.
  - `fnmr_list` (`list`): A list of FNMR values corresponding to each threshold.

- **Logic**:
  1. **Create DataFrame**: Constructs a DataFrame from the input lists, with columns for `Threshold`, `FMR`, and `FNMR`.
  2. **Set Index**: Sets the `Threshold` column as the index for easy reference.
  3. **Display**: Prints the table, allowing users to view FMR and FNMR values for each threshold.

---

# utils.py

This file has small straight-forward helper functions to help keep the rest of the code clean.

Here's a summary of each function's purpose:

`save_features(features, labels, filepath)`: This function saves a tuple containing features and labels to a specified file using the pickle module for serialization.

`load_features(filepath)`: This function loads the features and labels from a specified file, also using pickle.

`create_output_dir(name, type)`: This function creates a new output directory structure based on the provided name and type ('train' or 'test'). It checks if the directory already exists to avoid duplication.

`extract_labels(image_names)`: This function extracts numerical labels from a list of image names using regular expressions, returning them as a list of integers.

`load_images_from_folder(base_folder)`: This function loads images from specified training and testing folders within subject directories. It iterates through each subject's directory, looks for training images in a subfolder named '1' and testing images in '2', and adds the image paths to a dictionary categorized as 'train' and 'test'.

Overall, these functions are designed to facilitate the organization, storage, and retrieval of image data and their associated labels for further processing or analysis in a machine learning context.

## Limitation(s) of the current design:

Iris detection is not very smart due to the concentric circles assumption despite its performance being better than the threshold/edge detection/hough circles method.
Currently, enhancement is pretty basic and does not account for reflections or
eyelashes. Image enhancement also performed better just performing histogram equalization on the entire image rather than subtracting background illumination in 16x16 blocks and then histogram equalization on each 32x32 block. For feature extraction, we are limited to two sinusodal modulating functions, whereas introducing more functions could extract more features of the iris for iris matching. In addition, when reducing dimensionality using the Fisher Linear Discriminant, we can only represent each iris using the number of features as there are irises in the dataset. In order to have a better representation of the irises, it would be useful to have more data and thus a higher dimension feature vector to classify irises.

## Improvements:

We could make improvements on locating the iris using the same thresholding/Canny edge/Hough
circle method with better parameters or improved image processing to help with the noise.
Image enhancement could be further improved by accounting for reflections and eyelashes. Additionally finetuning the background illumination subtraction and then histogram equalization on the smaller blocks could be improved. Perhaps smaller blocks would perform better. In addition, adjusting the Region of Interest to make it slightly smaller, introducing more sinusodal functions into the spatial filter, augmenting our data to represent it in higher dimensions, and testing a larger set of rotation angles could be potential improvements to increase our CRR of the irises.

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
