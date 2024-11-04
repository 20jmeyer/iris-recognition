import cv2
import numpy as np
import sys
import IrisNormalization


def enhance_iris_basic(norm_iris):
    norm_iris = IrisNormalization.ensure_gray(norm_iris)
    # Perform histogram equalization to enhance the normalized iris
    enhanced_image = cv2.equalizeHist(norm_iris)
    return enhanced_image


def enhance_iris(norm_iris):
    norm_iris = IrisNormalization.ensure_gray(norm_iris)

    # Estimate background illumination by finding mean of 16x16 blocks
    background_est = np.zeros_like(norm_iris, dtype=np.float32)
    size = 16
    for y in range(0, norm_iris.shape[0], size):
        for x in range(0, norm_iris.shape[1], size):
            block = norm_iris[y:y + size, x:x + size]
            block_mean = np.mean(block)
            background_est[y:y + size, x:x + size] = block_mean

    # Expand using bicubic interpolation
    background_est = cv2.resize(background_est, (norm_iris.shape[1], norm_iris.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Subtract estimated background illumination from the normalized image
    light_compensated = cv2.subtract(norm_iris.astype(np.float32), background_est)
    light_compensated = np.clip(light_compensated, 0, 255).astype(np.uint8)

    # Enhance lighting corrected image through histogram equalization in each 32x32 region
    enhanced_image = np.zeros_like(light_compensated)
    size2 = 32
    for y in range(0, light_compensated.shape[0], size2):
        for x in range(0, light_compensated.shape[1], size2):
            block = light_compensated[y:y+size2, x:x+size2]
            block_equalized = cv2.equalizeHist(block)
            enhanced_image[y:y+size2, x:x+size2] = block_equalized

    return enhanced_image.astype(np.uint8)


def main():
    # Load isolated iris'
    mask_path = "./norm_output/train/024_1_1_iris.bmp"
    mask = cv2.imread(mask_path)

    # en = enhance_iris_basic(mask)
    en = enhance_iris(mask)

    cv2.imshow('Normalized iris', mask)
    cv2.imshow('Enhanced iris', en)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()