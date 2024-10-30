import cv2
import numpy as np
import math
from scipy.spatial import distance

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    print(image_path)
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_image = cv2.bilateralFilter(gray_image,9,75, 75)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0) #Helps to deal with eyelashes
    return image, gray_image

# Function to calculate projections and estimate pupil center
def estimate_pupil_center(image):
    horizontal_projection = np.sum(image, axis=1)  # Sum of rows
    vertical_projection = np.sum(image, axis=0)    # Sum of columns
    y_center = np.argmin(horizontal_projection)      # Y coordinate of initial estimated pupil center
    x_center = np.argmin(vertical_projection)        # X coordinate of initial estimated pupil center
    return (x_center, y_center)

def adjust_gamma(image, gamma=1.5):
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def darken_blacks(image, threshold_value=50, darkening_factor=0.5):
    # Create a mask for pixels below the threshold (i.e., dark pixels)
    mask = image < threshold_value

    # Darken those pixels by reducing their intensity
    image[mask] = (image[mask] * darkening_factor).astype(np.uint8)
    
    return image


# Function to find a thresholded sub-image around the estimated center
# Function to find a thresholded sub-image around the estimated center
def find_thresholded_subimage(image, center, lower_bound, upper_bound, size=120, is_pupil=True, repeat=False):
    x_center, y_center = center
    half_size = size // 2
    
    # Get the top-left corner of the sub-image
    x_start = max(x_center - half_size, 0)
    y_start = max(y_center - half_size, 0)

    # Extract the sub-image
    subimage = image[y_start: y_start + size, x_start: x_start + size]
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    #subimage = cv2.bilateralFilter(subimage,9,75, 75)
    #cv2.imshow('?image', subimage)
    #cv2.waitKey(0)
    # Threshold the sub-image
    if is_pupil:
        if not repeat:
            subimage = cv2.GaussianBlur(subimage, (5, 5), 0) #Helps to deal with eyelashes
            #subimage = cv2.equalizeHist(subimage)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            subimage = clahe.apply(subimage)
            mean_brightness = np.mean(subimage)
            #print(mean_brightness)
            if mean_brightness >= 132:
                print("making darker")
                subimage = adjust_gamma(subimage, gamma=.6)
                """cv2.imshow('before', subimage)
                cv2.waitKey(0)
                cv2.destroyAllWindows()"""
                subimage = darken_blacks(subimage,50,.2)
                """cv2.imshow('after', subimage)
                cv2.waitKey(0)
                cv2.destroyAllWindows()"""
                #subimage = clahe.apply(subimage)
            
            # Apply CLAHE to the grayscale image
            #
            """cv2.imshow('imagepulil', subimage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
        ret, thresh = cv2.threshold(subimage,lower_bound,upper_bound,cv2.THRESH_BINARY)
        """thresh = cv2.adaptiveThreshold(subimage, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 
                                   blockSize=11, 
                                   C=2)"""
        kernel = np.ones((5, 5), np.uint8) 
        """cv2.imshow('imagepulil', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        iters = 3 if repeat else 1
        thresh = cv2.dilate(thresh, kernel, iterations=iters)
        """cv2.imshow('imagepulil', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        thresh = cv2.erode(thresh, kernel, iterations=iters)
        """cv2.imshow('imagepulil', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        """cv2.imshow('imagepulil', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
    else: 
        #subimage = cv2.GaussianBlur(subimage, (5, 5), 0)
        subimage = cv2.equalizeHist(subimage)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
        # Apply CLAHE to the grayscale image
        subimage = clahe.apply(subimage)
        ret, thresh = cv2.threshold(subimage,lower_bound,upper_bound,cv2.THRESH_BINARY)
        """thresh = cv2.adaptiveThreshold(subimage, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 
                                   blockSize=11, 
                                   C=2)"""
        """cv2.imshow('theesh', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
    
    # Calculate moments of the binary image
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]) + x_start
        cy = int(M["m01"] / M["m00"]) + y_start
    else:
        cx, cy = x_center, y_center

    return thresh, (cx, cy), (x_start, y_start)

# Function to draw circles on the image
def draw_circle(image, center, radius, color=(255, 244, 0), thickness=2):
    cv2.circle(image, center, radius, color, thickness)

def segment_pupil(image, pupil_circle):
    # Step 1: Create a mask for the pupil
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create a blank mask
    center_coordinates = (pupil_circle[0], pupil_circle[1])  # Center of the pupil
    radius = pupil_circle[2]  # Radius of the pupil
    cv2.circle(mask, center_coordinates, radius, (255), thickness=-1)  # Draw a filled circle (pupil mask)
    """cv2.imshow('maks', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    # Step 2: Apply the mask to the original image
    segmented_pupil = cv2.bitwise_not(image, image, mask=mask)  # Mask out the pupil area

    # Step 3: Show or return the segmented pupil image
    from matplotlib import pyplot as plt
    """plt.imshow(segmented_pupil, cmap='gray')  # Display the segmented pupil image
    plt.show()"""

    return segmented_pupil


# Update detect_iris and detect_pupil to return the top-left corner
def detect_iris(image, pupil_circle):
    center = estimate_pupil_center(image)
    attempt2, final_center, top_left_iris = find_thresholded_subimage(image, center, 100, 165, 250,False)
    #attempt2 = segment_pupil(attempt2, pupil_circle)
    masked_image = np.zeros_like(attempt2)
    height = attempt2.shape[0]
    masked_image[height // 2:, :] = attempt2[height // 2:, :]
    attempt2 = masked_image
    """cv2.imshow('att', attempt2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    mean, std = cv2.meanStdDev(attempt2)
    lower_threshold = max(0, mean[0] - std[0])  # Set a lower bound based on the mean and std
    upper_threshold = min(255, mean[0] + std[0])  # Set an upper bound similarly
    edges = cv2.Canny(attempt2, int(lower_threshold), int(upper_threshold))
    #edges = cv2.Canny(attempt2, 60, 110)
    from matplotlib import pyplot as plt
    """plt.imshow(edges)
    plt.show()"""
    contrast = np.std(edges)
    param1 = int(contrast * 1.5)  # Example adjustment
    param2 = max(20, int(contrast / 5))  # Ensure param2 is not too low
    circles =cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1.4,               # Finer resolution
        minDist=60,           # Minimum distance between circle centers
        param1=param1,           # Upper threshold for edge detection
        param2=param2,            # Accumulator threshold for circle centers
        minRadius=80,         # Minimum radius
        maxRadius=120         # Maximum radius
    )
    #circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 10, 90)
    
    circles = np.uint16(np.around(circles))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle on the edge-detected image
            draw_circle(edges, (i[0], i[1]), i[2], color=(255, 255, 0), thickness=2)  # Yellow color for the circle
            
            # Show the result
            """cv2.imshow('Detected Iris on Edges', edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
    best_circle = circles[0][0]  # Assuming there's only one circle
    # Adjust circle coordinates
    best_circle[0] += top_left_iris[0]
    best_circle[1] += top_left_iris[1]
    return best_circle, final_center

def pad_image(image, pad_size):
    # Add padding around the image
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def detect_pupil(image):
    center = estimate_pupil_center(image)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.equalizeHist(image)
    attempt1, center1, top_left_pupil = find_thresholded_subimage(image, center, 40, 250, 180)
    attempt1, center1, top_left_pupil = find_thresholded_subimage(image, center1, 40, 250, 120, True)
    """cv2.imshow('aa1', attempt1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    mean, std = cv2.meanStdDev(attempt1)
    lower_threshold = max(0, mean[0] - std[0])  # Set a lower bound based on the mean and std
    upper_threshold = min(255, mean[0] + std[0])  # Set an upper bound similarly
    edges = cv2.Canny(attempt1, int(lower_threshold), int(upper_threshold))
    from matplotlib import pyplot as plt
    edges = pad_image(edges, pad_size=50)
    """plt.imshow(edges)
    plt.show()"""
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
            
            # Show the result
            """cv2.imshow('Detected pupil on Edges', edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
    best_circle = circles[0][0]
    # Adjust circle coordinates
    best_circle[0] += top_left_pupil[0]-50 #need to undo the padding and the shrinking
    best_circle[1] += top_left_pupil[1]-50
    
    
    
    return best_circle, center1






# Main function to process the image and detect pupil and iris boundaries
def detect_pupil_and_iris(image_path):
    load_image, gray_image = load_and_preprocess_image(image_path)
    pupil_circle, pupil_center = detect_pupil(gray_image)
    iris_circle, iris_center = detect_iris(gray_image, pupil_circle)
    draw_circle(load_image, (int(iris_circle[0]), int(iris_circle[1])), int(iris_circle[2]))
    draw_circle(load_image, (int(pupil_circle[0]), int(pupil_circle[1])), int(pupil_circle[2]))
    
    
    # Show the final image with the detected circles
    #cv2.imshow('im2g', load_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
# Main function to locate the iris region between the iris and pupil circles
def locate_iris(image_path):
    # Load and preprocess the image
    print(image_path)
    load_image, gray_image = load_and_preprocess_image(image_path)
    
    # Detect the iris and pupil circles
    pupil_circle, pupil_center = detect_pupil(gray_image)
    iris_circle = naive_detect_iris(gray_image, pupil_circle)
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
    box_size = int(iris_circle[2] * 2.5)  # Example: double the iris radius
    x_center, y_center = pupil_circle[:2]

    # Calculate crop boundaries centered on the pupil
    x_start = max(int(x_center - box_size // 2), 0)
    y_start = max(int(y_center - box_size // 2), 0)
    x_end = min(x_start + box_size, load_image.shape[1])
    y_end = min(y_start + box_size, load_image.shape[0])

    # Crop the segmented image to this bounding box
    centered_segmented = segmented[y_start:y_end, x_start:x_end]
    """cv2.imshow('centered', centered_segmented)
    cv2.imshow('notcentered', segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    # Draw the detected circles for visualization
    draw_circle(load_image, (int(iris_circle[0]), int(iris_circle[1])), int(iris_circle[2]))
    draw_circle(load_image, (int(pupil_circle[0]), int(pupil_circle[1])), int(pupil_circle[2]))

    return centered_segmented, mask_between

def naive_detect_iris(image, pupil_circle):
    iris_radius = pupil_circle[2] + 53
    iris_circle = (pupil_circle[0], pupil_circle[1], iris_radius)
    return iris_circle
    

import cv2
import numpy as np
import matplotlib.pyplot as plt

def fit_parabola(points):
    if len(points) == 0:
        return None  # No points to fit
    x = points[:, 1]
    y = points[:, 0]

    # Fit a second-degree polynomial (parabola)
    coefficients = np.polyfit(x, y, 2)
    return coefficients  # Returns (a, b, c) for y = ax^2 + bx + c

def detect_eyelids(segmented_iris):
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
    # Show the masked image

# Example usage
# segmented_iris = cv2.imread('path_to_segmented_iris.jpg')
# detect_eyelids(segmented_iris)






locate_iris('./database/100/1/100_1_3.bmp')
