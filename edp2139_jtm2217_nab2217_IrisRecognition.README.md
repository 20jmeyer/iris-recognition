# iris-recognition

Replication of Iris detection and recognition paper.

## Design logic:

Implemented as described in _Personal Identification Based on Iris Texture Analysis_ by Ma et al.
using the CASIA Iris Image Database (version 1.0). Multiple iris images were first localized. Initially, we
tried thresholding subimages and then using Canny edge detection and Hough circles for both the pupil and
the iris. However we switched to just using this method for pupil detection due to better results with a
more naive method: For the iris, we instead naively assumed it is concentrically outside
the pupil by estimating the iris radius to be 53 pixels longer than the pupil's and used this
to find its bounding circle. Then, eyelids were detected using parabola fitting. A mask containing
only the isolated iris was made and we ensured this was cropped and centered. Next, came iris
normalization. The localized iris images were used as input. A mapping was made to transform the
circular iris shape in polar coordinates into a 64x512 rectangle in cartesian coordinates. Then,
the normalized iris images were enhanced. Enhancement was done using histogram equalization.

## Limitation(s) of the current design:

Iris detection is not very smart due to the concentric circles assumption. We tried the thresholding + canny edge + hough circles for the iris detection as well but the performance was not as good.

## Improvements:

We could make improvements on locating the iris using the same thresholding/Canny edge/Hough
circle method with better parameters or improved image processing to help with the noise.

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
