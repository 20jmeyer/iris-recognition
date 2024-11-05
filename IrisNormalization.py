import cv2
import sys
import numpy as np
import IrisLocalization  # for detect_pupil & naive_detect_iris


def ensure_gray(img):
    """
    Ensure image was loaded correctly and in grayscale.
    :param img: (nparray) image
    :return: (nparray) grayscale image
    """
    if img is None:
        print("Error loading image. Check the path.")
        sys.exit(1)
    else:
        if len(img.shape) == 3:
            gray_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_mask = img  # If already grayscale
    return gray_mask


def calc_boundary_point(theta, circle):
    """
    Calculate boundary point (x, y) at a given angle in circle.
    :param theta: (numpy.float64) given angle in radians
    :param circle: (list) circle to calculate points from (x_coord, y_coord, radius)
    :return: boundary points
    """
    bound_x = circle[0] + circle[2] * np.cos(theta)
    bound_y = circle[1] + circle[2] * np.sin(theta)
    return bound_x, bound_y


def normalize_iris(mask):
    """
    Map iris from Cartesian coordinates to polar coordinates.
    :param mask: (nparray) localized iris mask image
    :return: (nparray) normalized iris image as type uint8
    """
    gray_mask = ensure_gray(mask)

    # Find iris and pupil boundaries (naively), circle info saved as: x_coord, y_coord, radius
    pupil_circle, _ = IrisLocalization.detect_pupil(gray_mask)  # aka inner_bound
    iris_circle = IrisLocalization.naive_detect_iris(pupil_circle)  # aka outer_bound

    # 64 x 512 based off paper, 64 'rows', 512 'columns'
    m = 64
    n = 512

    # Initialize empty array to store normalization,
    normalized_iris = np.zeros((m, n))

    # Divide 360 degree polar coord space into n angle segments
    thetas = np.linspace(0, 2 * np.pi, n)

    # Iterate over each angle segment
    for x in range(n):
        theta = thetas[x]
        # Find inner and outer boundary points at this theta
        inner_x, inner_y = calc_boundary_point(theta, pupil_circle)  # may need to by type cast to int
        outer_x, outer_y = calc_boundary_point(theta, iris_circle)   # may need to by type cast to int

        # Interpolate points between the inner and outer boundary
        for y in range(m):
            interp_factor = y / m

            # Calculate cartesian coordinates of the normalized point
            interp_x = inner_x + (outer_x - inner_x) * interp_factor
            interp_y = inner_y + (outer_y - inner_y) * interp_factor

            # Map to the original image and assign to normalized image
            if 0 <= int(interp_y) < gray_mask.shape[0] and 0 <= int(interp_x) < gray_mask.shape[1]:  # Ensure OG image has these coords
                normalized_iris[y, x] = gray_mask[int(interp_y), int(interp_x)]

    return normalized_iris.astype(np.uint8)


def main():
    # Load isolated iris
    mask_path = "./localized_output/train/003_1_1_iris.bmp"
    mask = cv2.imread(mask_path)

    norm_iris = normalize_iris(mask)

    cv2.imshow('Localized iris', mask)
    cv2.imshow('Normalized iris',norm_iris)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
