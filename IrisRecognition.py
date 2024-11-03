import os
import re
import cv2
import numpy as np
import IrisLocalization
import IrisNormalization
import ImageEnhancement
import FeatureExtraction
import IrisMatching 


def load_images_from_folder(base_folder):
    images = {
        'train': [],
        'test': []
    }
    
    # Iterate over each subject folder in the base folder
    for subject in os.listdir(base_folder):
        subject_path = os.path.join(base_folder, subject)
        
        # Check if the subject path is a directory
        if os.path.isdir(subject_path):
            # paths for training and testing folders
            train_folder = os.path.join(subject_path, '1')
            test_folder = os.path.join(subject_path, '2')
            
            # training images
            if os.path.isdir(train_folder):
                for image_name in os.listdir(train_folder):
                    if image_name.endswith('.bmp'):
                        #print(image_name)
                        image_path = os.path.join(train_folder, image_name)
                        image = cv2.imread(image_path)
                        images['train'].append(image_path)
            
            # testing images
            if os.path.isdir(test_folder):
                for image_name in os.listdir(test_folder):
                    if image_name.endswith('.bmp'):
                        image_path = os.path.join(test_folder, image_name)
                        image = cv2.imread(image_path)
                        images['test'].append(image_path)
    
    return images


def create_output_dir(name, type):
    """
    Creates new output directories if they don't already exist
    :param name: (str) Name of the new output folder
    :param type: (str) 'train' or 'test
    :return: None
    """
    path = f'./{name}/{type}'
    if not os.path.exists(path):
        os.makedirs(path)


def extract_labels(image_names):
    """
    Returns the label values given the image names
    """
    return [int(re.findall('\d{3}', x)[0]) for x in image_names]


def main():
    base_folder = './database'  # Need to update to the correct path
    images = load_images_from_folder(base_folder)

    # Access training and testing images
    train_images = images['train']
    test_images = images['test']

    # Extract the labels from the train and test images using the file names
    train_labels = extract_labels(train_images)
    test_labels = extract_labels(test_images)

    create_output_dir("localized_output", "train")
    create_output_dir("localized_output", "test")
    create_output_dir("norm_output", "train")
    create_output_dir("norm_output", "test")
    create_output_dir("enhanced_output", "train")
    create_output_dir("enhanced_output", "test")
    
    train_features = []
    for image in train_images:
        # Localization
        iris, _ = IrisLocalization.locate_iris(image)
        save_name =  './localized_output/train/'+os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(save_name, iris)

        # Normalization
        norm_iris = IrisNormalization.normalize_iris(iris)
        norm_name = './norm_output/train/' + os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(norm_name, norm_iris)

        # Enhancement
        enhanced_iris = ImageEnhancement.enhance_iris(norm_iris)
        enhanced_name = './enhanced_output/train/' + os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(enhanced_name, enhanced_iris)

        # Feature Extraction
        extracted_features = FeatureExtraction.feature_iris(enhanced_iris)
        train_features.append(extracted_features)

    # Fitting the dimension reduction model
    train_features = np.array(train_features)
    model, class_centers = IrisMatching.reduce_dimensionality(
        train_features, train_labels, n_components=100
    )

    # Match Iris for the 3 distance measures
    for i, feature in enumerate(train_features):

        L1_train_features, L1_train_reduced_features, L1_train_class, L1_train_probability = IrisMatching.match_iris(
            feature, class_centers, model, distance_type='L2'
        )

        L2_train_features, L2_train_reduced_features, L2train_class, L2_train_probability = IrisMatching.match_iris(
            feature, class_centers, model, distance_type='L2'
        )

        cosine_train_features, cosine_train_reduced_features, cosine_train_class, cosine_train_probability = IrisMatching.match_iris(
            feature, class_centers, model, distance_type='L2'
        )


    test_features = []
    for i, image in enumerate(test_images):
        # Localization
        iris, _ = IrisLocalization.locate_iris(image)
        save_name =  './localized_output/test/'+os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(save_name, iris)

        # Normalization
        norm_iris = IrisNormalization.normalize_iris(iris)
        norm_name = './norm_output/test/' + os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(norm_name, norm_iris)

        # Enhancement
        enhanced_iris = ImageEnhancement.enhance_iris(norm_iris)
        enhanced_name = './enhanced_output/test/' + os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(enhanced_name, enhanced_iris)

        # Feature Extraction
        extracted_features = FeatureExtraction.feature_iris(enhanced_iris)
        test_features.append(extracted_features)


    # Match Iris for the 3 distance measures
    test_features = np.array(test_features)
    for i, feature in enumerate(test_features):

        L1_test_features, L1_test_reduced_features, L1_test_class, L1_test_probability = IrisMatching.match_iris(
            feature, class_centers, model, distance_type='L1'
        )

        L2_test_features, L2_test_reduced_features, L2_test_class, L2_test_probability = IrisMatching.match_iris(
            feature, class_centers, model, distance_type='L2'
        )

        cosine_test_features, cosine_test_reduced_features, cosine_test_class, cosine_test_probability = IrisMatching.match_iris(
            feature, class_centers, model, distance_type='cosine'
        )

main()


