import numpy as np
import cv2

def spatial_filter(x, y, sigma_x, sigma_y, f):
    """
    Defines a spatial filter based on the Gaussian envelope and modulating functions.
    
    Args:
        x (float): X-coordinate of the spatial filter grid.
        y (float): Y-coordinate of the spatial filter grid.
        sigma_x (float): Standard deviation along the x-axis for the Gaussian envelope.
        sigma_y (float): Standard deviation along the y-axis for the Gaussian envelope.
        f (float): Frequency of the modulating function.
    Returns:
        float: The computed filter value at (x, y)
    """

    # The Gaussian envelope, as per the paper
    G = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-0.5 * ((x**2 / sigma_x**2) + (y**2 / sigma_y**2)))

    # The defined modulation function, as per the paper
    M1 = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))

    return G * M1


def apply_spatial_filter(image, sigma_x, sigma_y, f):
    """
    Applies a spatial filter to the image to extract frequency information.

    Args:
        image (np.ndarray): Grayscale input image of the Region of Interest (ROI).
        sigma_x (float): Standard deviation along the x-axis for the Gaussian envelope.
        sigma_y (float): Standard deviation along the y-axis for the Gaussian envelope.
        f (float): Frequency of the modulating function.

    Returns:
        np.ndarray: Filtered image with frequency information.
    """
    rows, cols = image.shape

    # Center the spatial filter at each pixel, and create a new array the same size as the original image
    filter_kernel = np.fromfunction(lambda x, y: spatial_filter(x - rows//2, y - cols//2, sigma_x, sigma_y, f), (rows, cols))

    # Convolve the centered filter with the image
    filtered_image = cv2.filter2D(image, -1, filter_kernel)
    return filtered_image


def extract_features(filtered_image):
    """
    Extracts the mean and absolute deviation from each 8x8 patch of the filtered image.

    Args:
        filtered_image (np.ndarray): Filtered image of the ROI.

    Returns:
        np.ndarray: A 1D array of feature values containing mean and absolute deviation for each block.
    """
    features = []
    patch_size = 8

    # Loop through the "rows" and "columns" of the image to make image patches
    for i in range(0, filtered_image.shape[0], patch_size):
        for j in range(0, filtered_image.shape[1], patch_size):
            patch = filtered_image[i:(i + patch_size), j:(j + patch_size)]

            # Calculate the mean and absolute deviation of each 8x8 image patch
            patch_mean = np.mean(np.abs(patch))
            patch_deviation = np.mean(np.abs(patch - patch_mean))
            features.extend([patch_mean, patch_deviation])

    return np.array(features)


def feature_iris(enhanced_iris):
    """
    Puts all of the steps together to apply a spatial feature.

    Args:
        enhanced_iris (np.ndarray): Grayscale enhanced iris image (output of the enhancement function).

    Returns:
        np.ndarray: A 1D feature vector (1536, ) containing frequency information.
    """
    # ROI of the image
    roi = enhanced_iris[:48, :] 

    # Apply the spatial filters in two domains
    filtered_image_1 = apply_spatial_filter(roi, sigma_x=3, sigma_y=1.5, f=1/4)
    filtered_image_2 = apply_spatial_filter(roi, sigma_x=4.5, sigma_y=1.5, f=1/16)

    # After filtering, extract the feature vectors
    features_1 = extract_features(filtered_image_1)
    features_2 = extract_features(filtered_image_2)
    feature_vector = np.concatenate((features_1, features_2))

    return feature_vector
