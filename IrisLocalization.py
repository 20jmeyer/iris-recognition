import cv2
import numpy as np
import math
from scipy.spatial import distance

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image

# Function to calculate projections and estimate pupil center
def estimate_pupil_center(image):
    horizontal_projection = np.sum(image, axis=1)  # Sum of rows
    vertical_projection = np.sum(image, axis=0)    # Sum of columns
    y_center = np.argmin(horizontal_projection)      # Y coordinate of initial estimated pupil center
    x_center = np.argmin(vertical_projection)        # X coordinate of initial estimated pupil center
    return (x_center, y_center)

# Function to find a thresholded sub-image around the estimated center
def find_thresholded_subimage(image, center, size=120):
    x_center, y_center = center
    half_size = size // 2
    
    # Extract a sub-image centered at the estimated pupil center
    subimage = image[y_center - half_size: y_center + half_size, 
                     x_center - half_size: x_center + half_size]
    
    # Threshold the sub-image to create a binary image
    ret, thresh = cv2.threshold(subimage, 50, 255, cv2.THRESH_BINARY)
    
    # Calculate moments of the binary image
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]) + max(x_center - half_size, 0)
        cy = int(M["m01"] / M["m00"]) + max(y_center - half_size, 0)
    else:
        cx, cy = x_center, y_center  # Default to the original center if no mass found
        
    return thresh, (cx, cy)

# Function to draw circles on the image
def draw_circle(image, center, radius, color=(255, 244, 0), thickness=2):
    cv2.circle(image, center, radius, color, thickness)

# Main function to process the image and detect pupil and iris boundaries
def detect_pupil_and_iris(image_path):
    load_image, gray_image = load_and_preprocess_image(image_path)

    # Estimate the pupil center
    center = estimate_pupil_center(gray_image)

    # Perform the first and second thresholded subimages
    attempt1, center1 = find_thresholded_subimage(gray_image, center)
    attempt2, final_center = find_thresholded_subimage(gray_image, center1)

    # Edge detection and circle detection
    edges = cv2.Canny(attempt2, 10, 100)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 10, 60)
    circles = np.around(circles)

    # Find the best circle based on the minimum distance to the final center
    best_circle = None
    min_distance = math.inf
    for circle in circles[0]:
        circle_center = (circle[0], circle[1])
        cur_distance = distance.euclidean(final_center, circle_center)
        if cur_distance < min_distance: 
            min_distance = cur_distance
            best_circle = circle

    # Print the best circle's parameters
    print(best_circle)

    # Calculate the adjusted circle coordinates
    circle_x = int(best_circle[0]) + (center1[0] - (120 // 2))  # Adjust X position
    circle_y = int(best_circle[1]) + (center1[1] - (120 // 2))  # Adjust Y position
    radius = int(best_circle[2])

    # Draw the best circle on the original image
    draw_circle(load_image, (circle_x, circle_y), radius)
    draw_circle(load_image, (circle_x, circle_y), radius + 60)

    # Show the final image with the detected circles
    cv2.imshow('img', load_image)
    cv2.waitKey(0)

# Run the detection function with the image path
detect_pupil_and_iris('./database/001/1/001_1_1.bmp')
