import cv2
import sys
import IrisNormalization

def enhance_iris(norm_iris):
    norm_iris = IrisNormalization.ensure_gray(norm_iris)
    # Perform histogram equalization to enhance the normalized iris
    enhanced_image = cv2.equalizeHist(norm_iris)
    return enhanced_image

def main():
    # Load isolated iris
    mask_path = "./norm_output/train/024_1_1_iris.bmp"
    mask = cv2.imread(mask_path)

    en = enhance_iris(mask)

    cv2.imshow('Normalized iris', mask)
    cv2.imshow('Enhanced iris', en)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()