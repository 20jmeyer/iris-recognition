import os
import re
import cv2
import numpy as np
import pandas as pd
import IrisLocalization
import IrisNormalization
import ImageEnhancement
import FeatureExtraction
import IrisMatching 
import PerformanceEvaluation


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
        train_features, train_labels, n_components=None
    )

    reduced_model, reduced_class_centers = IrisMatching.reduce_dimensionality(
        train_features, train_labels, n_components=100
    )

    # Match Iris for the 3 distance measures
    for i, feature in enumerate(train_features):

        L1_train_features, L1_train_class, L1_train_probability, L1_train_features_reduced, \
        L1_train_class_reduced, L1_train_probability_reduced = IrisMatching.match_iris(
            feature, class_centers, reduced_class_centers,reduced_model, distance_type='L1'
        )

        L2_train_features, L2_train_class, L2_train_probability, L2_train_features_reduced, \
        L2_train_class_reduced, L2_train_probability_reduced = IrisMatching.match_iris(
            feature, class_centers, reduced_class_centers, reduced_model, distance_type='L2'
        )

        cosine_train_features, cosine_train_class, cosine_train_probability, cosine_train_features_reduced, \
        cosine_train_class_reduced, cosine_train_probability_reduced = IrisMatching.match_iris(
            feature, class_centers, reduced_class_centers, reduced_model, distance_type='cosine'
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
        
    thresholds = [0.446, 0.472, 0.502]
    fmr = []
    fnmr = []
    dimensions = [30, 60, 80, 100, 107] #Different number of dimensions to try

    # Match Iris for the 3 distance measures
    test_features = np.array(test_features)

    CRR_RESULTS = []
    for dimension_target in dimensions:
        
        reduced_model, reduced_class_centers = IrisMatching.reduce_dimensionality(
            train_features, train_labels, n_components=dimension_target
        )

        crr_l1 = []
        crr_l2 = []
        crr_cosine = []


        for i, feature in enumerate(test_features):
            
            L1_test_features, L1_test_class, L1_test_probability, \
            L1_test_features_reduced, L1_test_class_reduced, L1_test_probability_reduced = IrisMatching.match_iris(
                feature, class_centers, reduced_class_centers, reduced_model, distance_type='L1'
            )
            crr_l1.append(L1_test_class_reduced)

            L2_test_features, L2_test_class, L2_test_probability, L2_test_features_reduced, \
            L2_test_class_reduced, L2_test_probability_reduced = IrisMatching.match_iris(
                feature, class_centers, reduced_class_centers, reduced_model, distance_type='L2'
            )
            crr_l2.append(L2_test_class_reduced)
            
            
            cosine_test_features, cosine_test_class, cosine_test_probability, cosine_test_features_reduced, \
            cosine_test_class_reduced, cosine_test_probability_reduced = IrisMatching.match_iris(
                feature, class_centers, reduced_class_centers, reduced_model, distance_type='cosine'
            )
            crr_cosine.append(cosine_test_class_reduced)
        
        crr_l1 = PerformanceEvaluation.CRR(crr_l1, test_labels)
        crr_l2 = PerformanceEvaluation.CRR(crr_l2, test_labels)
        crr_cosine = PerformanceEvaluation.CRR(crr_cosine, test_labels)
        CRR_RESULTS.append([crr_l1, crr_l2, crr_cosine])

    CRR_RESULTS = pd.DataFrame(CRR_RESULTS, columns=['crr_l1', 'crr_l2', 'crr_cosine'])
    CRR_RESULTS['dimensions'] = dimensions
    CRR_RESULTS = CRR_RESULTS.set_index('dimensions')

    PerformanceEvaluation.plot_CRR_curves(CRR_RESULTS)
main()


