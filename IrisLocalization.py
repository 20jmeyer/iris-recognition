import cv2
import numpy as np
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0) #Helps to deal with eyelashes
    return image, gray_image

# Function to calculate projections and estimate pupil center
def estimate_pupil_center(image):
    horizontal_projection = np.sum(image, axis=1)  # Sum of rows
    vertical_projection = np.sum(image, axis=0)    # Sum of columns
    y_center = np.argmin(horizontal_projection)      # Y coordinate of initial estimated pupil center
    x_center = np.argmin(vertical_projection)        # X coordinate of initial estimated pupil center
    return (x_center, y_center)

#Function to adjust gamma (I used this to darken certain parts of the image to help with thresholding)
def adjust_gamma(image, gamma=1.5):
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

#Another function to try to darken certain parts of the image (used this to help with pupil thresholding)
def darken_blacks(image, threshold_value=50, darkening_factor=0.5):
    # Create a mask for pixels below the threshold (i.e., dark pixels)
    mask = image < threshold_value

    # Darken those pixels by reducing their intensity
    image[mask] = (image[mask] * darkening_factor).astype(np.uint8)
    
    return image


# Function to find a thresholded sub-image around the estimated center
def find_thresholded_subimage(image, center, lower_bound, upper_bound, size=120, is_pupil=True, repeat=False):
    """
    Find a thresholded sub-image around the estimated center.

    This function extracts a square sub-image from the given image based on the specified center coordinates.
    It applies various image processing techniques, including Gaussian blur, contrast-limited adaptive histogram equalization (CLAHE),
    and binary thresholding, to enhance and threshold the sub-image. Morphological operations are also applied to refine the binary image.

    Parameters:
    - image (numpy.ndarray): The input image from which the sub-image will be extracted. 
      It is expected to be in grayscale format.
    - center (tuple): A tuple containing the (x, y) coordinates of the estimated center around which 
      the sub-image will be extracted.
    - lower_bound (int): The lower threshold value for binary thresholding.
    - upper_bound (int): The upper threshold value for binary thresholding.
    - size (int, optional): The size of the square sub-image to be extracted. Defaults to 120 pixels.
    - is_pupil (bool, optional): A flag indicating whether the processing is for the pupil (True) or the iris (False).
      This affects the preprocessing steps applied to the sub-image. Defaults to True.
    - repeat (bool, optional): A flag indicating whether this is a second attempt at thresholding.
      If True, different morphological operations are applied to refine the thresholded image. Defaults to False.

    Returns:
    - thresh (numpy.ndarray): The binary thresholded sub-image.
    - (cx, cy) (tuple): A tuple containing the (x, y) coordinates of the centroid of the thresholded area.
    - (x_start, y_start) (tuple): A tuple containing the (x, y) coordinates of the top-left corner of the sub-image.

    Notes:
    - The function applies morphological operations to reduce the impact of eyelashes in the thresholded image 
      when processing the pupil. If the mean brightness of the sub-image is too high, it adjusts the gamma 
      to darken the image.
    - For iris processing, the sub-image is equalized before thresholding to enhance the features.
    """
    x_center, y_center = center
    half_size = size // 2
    
    # Get the top-left corner of the sub-image
    x_start = max(x_center - half_size, 0)
    y_start = max(y_center - half_size, 0)

    # Extract the sub-image
    subimage = image[y_start: y_start + size, x_start: x_start + size]

    # Threshold the sub-image
    if is_pupil:
        if not repeat: #The paper suggests to find a thesholded subimage twice, so the approach is slightly modified if we are on that second thresholding attempt
            subimage = cv2.GaussianBlur(subimage, (5, 5), 0) #Helps to deal with eyelashes
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #Function that helps increase contrast 
            subimage = clahe.apply(subimage)
            mean_brightness = np.mean(subimage) #Get average pixel intensity
            
            if mean_brightness >= 132: #If the image is too bright, darken it
                subimage = adjust_gamma(subimage, gamma=.6)
                subimage = darken_blacks(subimage,50,.2)

        ret, thresh = cv2.threshold(subimage,lower_bound,upper_bound,cv2.THRESH_BINARY) #Thesholds the image

        kernel = np.ones((5, 5), np.uint8) 

        iters = 3 if repeat else 1  #Morphological operations to try to get rid of the eyelashes
        thresh = cv2.dilate(thresh, kernel, iterations=iters)
        thresh = cv2.erode(thresh, kernel, iterations=iters)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    else: 
        subimage = cv2.equalizeHist(subimage)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
        # Apply CLAHE (contrast increaser) to the grayscale image
        subimage = clahe.apply(subimage)
        ret, thresh = cv2.threshold(subimage,lower_bound,upper_bound,cv2.THRESH_BINARY) #Thresholds the image
    
    # Calculating the centroid of the binary image
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]) + x_start
        cy = int(M["m01"] / M["m00"]) + y_start
    else:
        cx, cy = x_center, y_center

    return thresh, (cx, cy), (x_start, y_start) 

# Function to draw circles on the image
def draw_circle(image, center, radius, color=(255, 244, 0), thickness=2):
    """
    Draw a circle on an image.

    This function uses OpenCV to draw a circle on the specified image at the given center 
    with the specified radius, color, and thickness.

    Parameters:
    - image (numpy.ndarray): The image on which to draw the circle. It should be a valid 
      image array (e.g., a grayscale or color image).
    - center (tuple): A tuple containing the (x, y) coordinates of the center of the circle.
    - radius (int): The radius of the circle to be drawn.
    - color (tuple, optional): A tuple representing the color of the circle in BGR format. 
      Defaults to (255, 244, 0) which is a shade of yellow.
    - thickness (int, optional): The thickness of the circle outline. Defaults to 2. If 
      set to a negative value, the circle will be filled.

    Returns:
    - None: The function modifies the input image in place and does not return a value.
    """
    cv2.circle(image, center, radius, color, thickness)

def segment_pupil(image, pupil_circle):
    """
    Segment the pupil from an image using a circular mask.

    This function creates a mask for the pupil based on the provided pupil circle parameters 
    and applies it to the input image to isolate the pupil region.

    Parameters:
    - image (numpy.ndarray): The input image from which the pupil is to be segmented. 
      It should be a valid image array (e.g., grayscale or color image).
    - pupil_circle (tuple): A tuple containing the (x, y) coordinates of the center of the 
      pupil and its radius, structured as (x_center, y_center, radius).

    Returns:
    - segmented_pupil (numpy.ndarray): The image with the pupil region segmented out, 
      where the pupil area is inverted (black) and the rest of the image remains unchanged.
    """
    #Create a mask for the pupil
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create a blank mask
    center_coordinates = (pupil_circle[0], pupil_circle[1])  # Center of the pupil
    radius = pupil_circle[2]  # Radius of the pupil
    cv2.circle(mask, center_coordinates, radius, (255), thickness=-1)  # Draw a filled circle (pupil mask)

    # Apply the mask to the original image
    segmented_pupil = cv2.bitwise_not(image, image, mask=mask)  # Mask out the pupil area

    return segmented_pupil


# Update detect_iris and detect_pupil to return the top-left corner
def detect_iris(image, pupil_circle):
    """
    Detect the iris in a given image based on the estimated pupil center.

    This function estimates the pupil center, thresholds the image to isolate the iris region, 
    applies edge detection, and utilizes the Hough Transform to detect circular shapes that 
    correspond to the iris. It also returns the coordinates of the detected iris circle 
    along with the final pupil center.

    Parameters:
    - image (numpy.ndarray): The input image in which to detect the iris. 
      It should be a valid image array (e.g., grayscale or color image).
    - pupil_circle (tuple): A tuple containing the (x, y) coordinates of the center of the 
      pupil and its radius, structured as (x_center, y_center, radius).

    Returns:
    - best_circle (numpy.ndarray): A 1D array containing the (x, y, radius) of the detected iris circle.
    - final_center (tuple): A tuple representing the estimated center of the pupil (x, y).
    """
    center = estimate_pupil_center(image)
    attempt2, final_center, top_left_iris = find_thresholded_subimage(image, center, 100, 165, 250,False) #Thresholds the image
    #attempt2 = segment_pupil(attempt2, pupil_circle)
    masked_image = np.zeros_like(attempt2)
    height = attempt2.shape[0]
    masked_image[height // 2:, :] = attempt2[height // 2:, :] #Just use the bottom half of the image (helps to edge detection by removing eyelashes)
    attempt2 = masked_image

    mean, std = cv2.meanStdDev(attempt2)
    lower_threshold = max(0, mean[0] - std[0])  # Set a lower bound based on the mean and std
    upper_threshold = min(255, mean[0] + std[0])  # Set an upper bound similarly
    # 'lower_threshold': Lower bound for identifying weak edges
    # 'upper_threshold': Upper bound for identifying strong edges
    # The Canny algorithm uses these thresholds to determine which edges to keep    
    edges = cv2.Canny(attempt2, int(lower_threshold), int(upper_threshold)) #Gets the edges from the thresholded image

    contrast = np.std(edges)
    param1 = int(contrast * 1.5) 
    param2 = max(20, int(contrast / 5))  # Ensure param2 is not too low
    circles =cv2.HoughCircles( #Circle detection from the edges using the Hough Circles algorithm
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1.4,               # Finer resolution
        minDist=60,           # Minimum distance between circle centers
        param1=param1,           # Upper threshold for edge detection
        param2=param2,            # Accumulator threshold for circle centers
        minRadius=80,         # Minimum radius
        maxRadius=120         # Maximum radius
    )
    
    circles = np.uint16(np.around(circles))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle on the edge-detected image
            draw_circle(edges, (i[0], i[1]), i[2], color=(255, 255, 0), thickness=2)  # Yellow color for the circle
            
    best_circle = circles[0][0]  # Only want one circle
    # Adjust circle coordinates
    best_circle[0] += top_left_iris[0]
    best_circle[1] += top_left_iris[1]
    return best_circle, final_center

def pad_image(image, pad_size):
    """
    Add padding around the input image. I used this because for some images, the 
    thresholded image was mostly out of bounds. Because of this, the hough circle detection
    would error out since the circle wasn't allowed to be mostly out of bounds. 
    I used padding to allow the hough circle detection to draw a circle that would otherwise
    be out of bounds.

    This function creates a new image with a specified padding size added to all sides 
    of the original image. The padding is filled with a constant color, which defaults 
    to black.

    Parameters:
    image (numpy.ndarray): The input image to be padded. It should be a 2D or 3D array 
                           representing the image in grayscale or color format.
    pad_size (int): The size of the padding to be added to each side of the image. 
                    This value specifies the number of pixels to pad on the top, 
                    bottom, left, and right.

    Returns:
    numpy.ndarray: The padded image, which is a new image array with the specified 
                   padding added around the original image.
    """
    # Add padding around the image
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def detect_pupil(image):
    """
    Detect the pupil in the provided image and return its coordinates.

    This function processes the input image to identify the location of the pupil 
    using a series of image processing techniques including Gaussian blurring, 
    histogram equalization, and Hough Circle Transform. It generates a mask for the 
    pupil area and refines the detection through thresholding.

    Parameters:
    image (numpy.ndarray): The input image in which to detect the pupil. This should 
                           be a grayscale image represented as a 2D array.

    Returns:
    tuple: A tuple containing:
        - best_circle (numpy.ndarray): A 1D array with the (x, y) coordinates of the 
          detected pupil center and its radius, adjusted for padding.
        - center1 (tuple): The estimated center of the pupil from the second thresholding 
          attempt, represented as (x, y) coordinates.
    """
    center = estimate_pupil_center(image)
    image = cv2.GaussianBlur(image, (5, 5), 0) #Helps with ignoring eyelashes
    image = cv2.equalizeHist(image) #Increases contrast
    attempt1, center1, top_left_pupil = find_thresholded_subimage(image, center, 40, 250, 180) #Thesholded subimage #1
    attempt1, center1, top_left_pupil = find_thresholded_subimage(image, center1, 40, 250, 120, True) #Thresholded subimage #2

    mean, std = cv2.meanStdDev(attempt1)
    lower_threshold = max(0, mean[0] - std[0])  # Set a lower bound based on the mean and std
    upper_threshold = min(255, mean[0] + std[0])  # Set an upper bound similarly
    edges = cv2.Canny(attempt1, int(lower_threshold), int(upper_threshold))

    edges = pad_image(edges, pad_size=50)

    contrast = np.std(edges)
    param1 = max(1,int(contrast * 1.5))  # Example adjustment
    param2 = max(20, int(contrast / 5))  # Ensure param2 is not too low
    circles = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1.5,              # Decrease the resolution ratio to detect larger circles
        minDist=60,          # Adjust this to avoid detecting overlapping circles
        param1=param1,           # Canny edge detection high threshold
        param2=param2,           # Accumulator threshold for circle detection, lower to capture more
        minRadius=30,        # Increase this to avoid detecting very small circles
        maxRadius=120        # Adjust based on the expected pupil size
    )

    circles = np.around(np.around(circles))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle on the edge-detected image
            draw_circle(edges, (i[0], i[1]), i[2], color=(255, 255, 0), thickness=2)  # Yellow color for the circle
            

    best_circle = circles[0][0]
    # Adjust circle coordinates
    best_circle[0] += top_left_pupil[0]-50 #need to undo the padding and the shrinking from getting subimages
    best_circle[1] += top_left_pupil[1]-50
    
    return best_circle, center1

    
    
    
# Main function to locate the iris region between the iris and pupil circles
def locate_iris(image_path):
    """
    Locate the iris region in an image based on detected pupil and iris circles.

    This function loads an image, processes it to detect the pupil and iris, 
    and extracts the region between the detected circles. It also applies masks 
    to isolate the iris region from the pupil region, ensuring that eyelids are 
    removed from the segmented output.

    Parameters:
    image_path (str): The file path to the input image in which to locate the iris.

    Returns:
    tuple: A tuple containing:
        - centered_segmented (numpy.ndarray): The cropped image segment containing 
          the iris region, centered around the pupil.
        - mask_between (numpy.ndarray): A binary mask representing the area between 
          the detected iris and pupil circles, used for segmentation purposes.
    """
    # Load and preprocess the image
    load_image, gray_image = load_and_preprocess_image(image_path)
    
    # Detect the iris and pupil circles
    pupil_circle, pupil_center = detect_pupil(gray_image)
    iris_circle = naive_detect_iris(pupil_circle)
    #iris_circle, iris_center = detect_iris(gray_image, pupil_circle)
    
    # Create a mask for the iris
    iris_mask = np.zeros(gray_image.shape[:2], dtype=np.uint8)
    cv2.circle(iris_mask, (int(iris_circle[0]), int(iris_circle[1])), int(iris_circle[2]), 255, thickness=-1)

    # Create a mask for the pupil
    pupil_mask = np.zeros(gray_image.shape[:2], dtype=np.uint8)
    cv2.circle(pupil_mask, (int(pupil_circle[0]), int(pupil_circle[1])), int(pupil_circle[2]), 255, thickness=-1)

    # Subtract pupil mask from iris mask to get the desired region
    mask_between = cv2.subtract(iris_mask, pupil_mask)

    # Apply the mask to the original image to get the segmented region
    segmented = cv2.bitwise_and(load_image, load_image, mask=mask_between)
    segmented = detect_eyelids(segmented)  # Removes eyelids
    
    # Define the bounding box size around the pupil center
    box_size = int(iris_circle[2] * 2.5)  #2.5 times the iris radius
    x_center, y_center = pupil_circle[:2]

    # Calculate crop boundaries centered on the pupil
    x_start = max(int(x_center - box_size // 2), 0)
    y_start = max(int(y_center - box_size // 2), 0)
    x_end = min(x_start + box_size, load_image.shape[1])
    y_end = min(y_start + box_size, load_image.shape[0])

    # Crop the segmented image to this bounding box
    centered_segmented = segmented[y_start:y_end, x_start:x_end]

    # Draw the detected circles for visualization
    draw_circle(load_image, (int(iris_circle[0]), int(iris_circle[1])), int(iris_circle[2]))
    draw_circle(load_image, (int(pupil_circle[0]), int(pupil_circle[1])), int(pupil_circle[2]))

    return centered_segmented, mask_between

def naive_detect_iris(pupil_circle):
    """
    Estimate the iris circle based on the detected pupil circle.

    This function provides a naive approach to detect the iris by assuming 
    the iris is a concentric circle from the pupil and therefore
    a fixed offset to the radius of the pupil circle. The center of the iris 
    circle is taken to be the same as the pupil's center.

    Parameters:
    pupil_circle (tuple): A tuple (x, y, r) representing the center coordinates (x, y) 
                          and radius (r) of the detected pupil circle.

    Returns:
    tuple: A tuple (x, y, r_iris) representing the estimated center coordinates 
           (x, y) and radius (r_iris) of the iris circle.
    """
    iris_radius = pupil_circle[2] + 53
    iris_circle = (pupil_circle[0], pupil_circle[1], iris_radius)
    return iris_circle
    

def fit_parabola(points):
    """
    Fits a parabolic curve (second-degree polynomial) to a set of points.

    Given a set of points, this function calculates the coefficients of a 
    parabola of the form y = ax^2 + bx + c that best fits the points using
    least-squares.

    Parameters:
    points (numpy.ndarray): A 2D array of shape (n, 2), where each row represents 
                            a point with coordinates (y, x). The first column 
                            contains the y-coordinates, and the second column 
                            contains the x-coordinates.

    Returns:
    numpy.ndarray or None: An array containing the coefficients (a, b, c) of the 
                           parabolic equation y = ax^2 + bx + c. Returns None 
                           if the input array is empty.
    """
    if len(points) == 0:
        return None  # No points to fit
    x = points[:, 1]
    y = points[:, 0]

    # Fit a second-degree polynomial (parabola)
    coefficients = np.polyfit(x, y, 2)
    return coefficients  # Returns (a, b, c) for y = ax^2 + bx + c

def detect_eyelids(segmented_iris):
    """
    Detect and mask out the eyelid regions in a segmented iris image.

    This function detects the approximate positions of the upper and lower 
    eyelids in a segmented iris image. It applies a Gaussian blur to the image 
    to reduce noise (hopefully mitigating eyelashes), uses edge detection to 
    identify eyelid edges, and fits parabolic curves to the detected edges. 
    It then creates a mask to cover regions above and below the eyelids based
    on the fitted parabolas.

    Parameters:
    segmented_iris (numpy.ndarray): Grayscale image containing the segmented 
                                    iris region with eyelids potentially 
                                    visible in the upper and lower parts.

    Returns:
    numpy.ndarray: The input image with eyelid regions masked out, where regions 
                   above the upper eyelid and below the lower eyelid are 
                   blacked out to isolate the iris more effectively.
    """
    original_image = segmented_iris.copy()  # Keep a copy of the original image
    image = cv2.GaussianBlur(segmented_iris, (5, 5), 0)
    edges = cv2.Canny(image, 50, 150)

    height, width = edges.shape

    upper_half = edges[0:int(height/2), :]
    lower_half = edges[int(height/2):, :]
    upper_edge_points = np.column_stack(np.nonzero(upper_half))
    lower_edge_points = np.column_stack(np.nonzero(lower_half))

    # Fit parabolas to the eyelids if edge points exist
    parabola_upper = fit_parabola(upper_edge_points)
    parabola_lower = fit_parabola(lower_edge_points)

    # Create a mask with the same size as the original image
    mask = np.ones((height, width), dtype=np.uint8) * 255  # Initialize mask with white

    # Check if the upper eyelid parabola was found
    if parabola_upper is not None:
        a_upper, b_upper, c_upper = parabola_upper
        for x in range(width):
            y_upper = int(a_upper * x**2 + b_upper * x + c_upper)  # Upper eyelid parabola
            if 0 <= y_upper < height // 2:
                mask[0:y_upper, x] = 0  # Mask out region above upper eyelid

    # Check if the lower eyelid parabola was found
    if parabola_lower is not None:
        a_lower, b_lower, c_lower = parabola_lower
        for x in range(width):
            y_lower = int(a_lower * x**2 + b_lower * x + c_lower)  # Lower eyelid parabola
            if height // 2 <= y_lower < height:
                mask[y_lower:, x] = 0  # Mask out region below lower eyelid

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    return masked_image



