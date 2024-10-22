import cv2
import numpy as np
import math
from scipy.spatial import distance

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    print(image_path)
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bilateralFilter(gray_image,9,75, 75)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0) #Helps to deal with eyelashes
    return image, gray_image

# Function to calculate projections and estimate pupil center
def estimate_pupil_center(image):
    horizontal_projection = np.sum(image, axis=1)  # Sum of rows
    vertical_projection = np.sum(image, axis=0)    # Sum of columns
    y_center = np.argmin(horizontal_projection)      # Y coordinate of initial estimated pupil center
    x_center = np.argmin(vertical_projection)        # X coordinate of initial estimated pupil center
    return (x_center, y_center)

# Function to find a thresholded sub-image around the estimated center
# Function to find a thresholded sub-image around the estimated center
def find_thresholded_subimage(image, center, lower_bound, upper_bound, size=120, is_pupil=True):
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
        subimage = cv2.GaussianBlur(subimage, (5, 5), 0) #Helps to deal with eyelashes
        thresh = cv2.adaptiveThreshold(subimage, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 
                                   blockSize=11, 
                                   C=2)
        #('image', thresh)
        #cv2.waitKey(0)
    else: 
        subimage = cv2.GaussianBlur(subimage, (5, 5), 0)
        subimage = cv2.equalizeHist(subimage)
        thresh = cv2.adaptiveThreshold(subimage, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 
                                   blockSize=11, 
                                   C=2)
    
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

# Update detect_iris and detect_pupil to return the top-left corner
def detect_iris(image):
    center = estimate_pupil_center(image)
    attempt2, final_center, top_left_iris = find_thresholded_subimage(image, center, 160, 180, 240,False)
    mean, std = cv2.meanStdDev(attempt2)
    lower_threshold = max(0, mean[0] - std[0])  # Set a lower bound based on the mean and std
    upper_threshold = min(255, mean[0] + std[0])  # Set an upper bound similarly
    edges = cv2.Canny(attempt2, int(lower_threshold), int(upper_threshold))
    #edges = cv2.Canny(attempt2, 60, 110)
    from matplotlib import pyplot as plt
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

def detect_pupil(image):
    center = estimate_pupil_center(image)
    attempt1, center1, top_left_pupil = find_thresholded_subimage(image, center, 90, 255, 140)
    mean, std = cv2.meanStdDev(attempt1)
    lower_threshold = max(0, mean[0] - std[0])  # Set a lower bound based on the mean and std
    upper_threshold = min(255, mean[0] + std[0])  # Set an upper bound similarly
    edges = cv2.Canny(attempt1, int(lower_threshold), int(upper_threshold))
    from matplotlib import pyplot as plt
    #plt.imshow(edges)
    #plt.show()
    contrast = np.std(edges)
    param1 = int(contrast * 1.5)  # Example adjustment
    param2 = max(20, int(contrast / 5))  # Ensure param2 is not too low
    circles = cv2.HoughCircles(
        attempt1, 
        cv2.HOUGH_GRADIENT, 
        dp=1.5,              # Decrease the resolution ratio to detect larger circles
        minDist=80,          # Adjust this to avoid detecting overlapping circles
        param1=param1,           # Canny edge detection high threshold
        param2=param2,           # Accumulator threshold for circle detection, lower to capture more
        minRadius=30,        # Increase this to avoid detecting very small circles
        maxRadius=120        # Adjust based on the expected pupil size
    )

    circles = np.around(np.around(circles))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        """for i in circles[0, :]:
            # Draw the outer circle on the edge-detected image
            draw_circle(edges, (i[0], i[1]), i[2], color=(255, 255, 0), thickness=2)  # Yellow color for the circle
            
            # Show the result
            cv2.imshow('Detected pupil on Edges', edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
    best_circle = circles[0][0]
    # Adjust circle coordinates
    best_circle[0] += top_left_pupil[0]
    best_circle[1] += top_left_pupil[1]
    return best_circle, center1






# Main function to process the image and detect pupil and iris boundaries
def detect_pupil_and_iris(image_path):
    load_image, gray_image = load_and_preprocess_image(image_path)
    iris_circle, iris_center = detect_iris(gray_image)
    pupil_circle, pupil_center = detect_pupil(gray_image)
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
    """cv2.imshow('img', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    # Detect the iris and pupil circles
    iris_circle, iris_center = detect_iris(gray_image)
    pupil_circle, pupil_center = detect_pupil(gray_image)
    
    # Create a mask for the iris
    iris_mask = np.zeros(gray_image.shape[:2], dtype=np.uint8)
    cv2.circle(iris_mask, (int(iris_circle[0]), int(iris_circle[1])), int(iris_circle[2]), 255, thickness=-1)  # White circle for iris

    # Create a mask for the pupil
    pupil_mask = np.zeros(gray_image.shape[:2], dtype=np.uint8)
    cv2.circle(pupil_mask, (int(pupil_circle[0]), int(pupil_circle[1])), int(pupil_circle[2]), 255, thickness=-1)  # White circle for pupil

    # Subtract pupil mask from iris mask to get the desired region
    mask_between = cv2.subtract(iris_mask, pupil_mask)

    # Apply the mask to the original image to get the segmented region
    segmented = cv2.bitwise_and(load_image, load_image, mask=mask_between)

    # Draw the detected circles for visualization
    draw_circle(load_image, (int(iris_circle[0]), int(iris_circle[1])), int(iris_circle[2]))
    draw_circle(load_image, (int(pupil_circle[0]), int(pupil_circle[1])), int(pupil_circle[2]))

    # Show the segmented region between the iris and pupil
    #cv2.imshow('Segmented Region Between Iris and Pupil', segmented)
    
    # Show the original image with detected circles
    #cv2.imshow('Detected Circles', load_image)
    
    # Wait for a key press and close windows
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    """cv2.imshow('segmented', segmented)
    cv2.waitKey()
    cv2.destroyAllWindows()"""
    return segmented, mask_between

#locate_iris('./database/104/2/104_2_4.bmp')
