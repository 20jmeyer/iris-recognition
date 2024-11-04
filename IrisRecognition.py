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

from utils import extract_labels,create_output_dir,load_features,save_features,load_images_from_folder

def main():
    base_folder = './database'  # Need to update to the correct path
    train_features_path = './train_features.pkl'
    test_features_path = './test_features.pkl'
    print("Running iris recognition...")
    ###
    ### To save time, I added the option to just load the features
    ###
    if os.path.exists(train_features_path) and os.path.exists(test_features_path): #set this to false to process all images
        # Load saved features
        print("Loading features from saved files...")
        train_features, train_labels = load_features(train_features_path)
        test_features, test_labels = load_features(test_features_path)
    else:
        print("Processing images...")   
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
            print("Normalizing images...")
            norm_iris = IrisNormalization.normalize_iris(iris)
            norm_name = './norm_output/train/' + os.path.basename(image)[:-4] + '_iris.bmp'
            cv2.imwrite(norm_name, norm_iris)

            # Enhancement
            print("Enhancing images...")
            enhanced_iris = ImageEnhancement.enhance_iris(norm_iris)
            enhanced_name = './enhanced_output/train/' + os.path.basename(image)[:-4] + '_iris.bmp'
            cv2.imwrite(enhanced_name, enhanced_iris)

            # Feature Extraction
            print("Extracting features...")
            extracted_features = FeatureExtraction.feature_iris(enhanced_iris)
            train_features.append(extracted_features)

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

        train_features = np.array(train_features)
        test_features = np.array(test_features)
        save_features(train_features, train_labels, train_features_path)
        save_features(test_features, test_labels, test_features_path)


    # Fitting the dimension reduction model
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
            feature, class_centers, reduced_class_centers, reduced_model, distance_type='L1'
        )

        L2_train_features, L2_train_class, L2_train_probability, L2_train_features_reduced, \
        L2_train_class_reduced, L2_train_probability_reduced = IrisMatching.match_iris(
            feature, class_centers, reduced_class_centers, reduced_model, distance_type='L2'
        )

        cosine_train_features, cosine_train_class, cosine_train_probability, cosine_train_features_reduced, \
        cosine_train_class_reduced, cosine_train_probability_reduced = IrisMatching.match_iris(
            feature, class_centers, reduced_class_centers, reduced_model, distance_type='cosine'
        )
    
    dimensions = [30, 60, 80, 100, 107] #Different number of dimensions to try

    CRR_RESULTS = {
        'Number of Dimensions': [],
        'crr_l1_normal': [],
        'crr_l1_reduced': [],
        'crr_l2_normal': [],
        'crr_l2_reduced': [],
        'crr_cosine_normal': [],
        'crr_cosine_reduced': []
    }
    
    COSINE_SIMILARITY = None
    COSINE_PREDS = None

    
    for dimension_target in dimensions:
        print(f"Matching with {dimension_target} dimensions...")
        reduced_model, reduced_class_centers = IrisMatching.reduce_dimensionality(
            train_features, train_labels, n_components=dimension_target
        )

        crr_l1_normal = []
        crr_l1_reduced = []
        crr_l2_normal = []
        crr_l2_reduced = []
        crr_cosine_normal = []
        crr_cosine_reduced = []

        cosine_similarity = []
        cosine_preds = []

        for i, feature in enumerate(test_features):
            
            L1_test_features, L1_test_class, L1_test_similarity, \
            L1_test_features_reduced, L1_test_class_reduced, L1_test_similarity_reduced = IrisMatching.match_iris(
                feature, class_centers, reduced_class_centers, reduced_model, distance_type='L1'
            )
            crr_l1_normal.append(L1_test_class)
            crr_l1_reduced.append(L1_test_class_reduced)

            L2_test_features, L2_test_class, L2_test_similarity, L2_test_features_reduced, \
            L2_test_class_reduced, L2_test_similarity_reduced = IrisMatching.match_iris(
                feature, class_centers, reduced_class_centers, reduced_model, distance_type='L2'
            )
            crr_l2_normal.append(L2_test_class)
            crr_l2_reduced.append(L2_test_class_reduced)
            
            cosine_test_features, cosine_test_class, cosine_test_similarity, cosine_test_features_reduced, \
            cosine_test_class_reduced, cosine_test_similarity_reduced = IrisMatching.match_iris(
                feature, class_centers, reduced_class_centers, reduced_model, distance_type='cosine'
            )
            
            crr_cosine_normal.append(cosine_test_class)
            crr_cosine_reduced.append(cosine_test_class_reduced)

            cosine_similarity.append(cosine_test_similarity)
            cosine_preds.append(cosine_test_class)
            
            
        print("Calculating CRR...")
        # Calculating the correct recognition rate for each type of similarity
        crr_l1_normal_rate = PerformanceEvaluation.CRR(crr_l1_normal, test_labels)
        crr_l1_reduced_rate = PerformanceEvaluation.CRR(crr_l1_reduced, test_labels)
        crr_l2_normal_rate = PerformanceEvaluation.CRR(crr_l2_normal, test_labels)
        crr_l2_reduced_rate = PerformanceEvaluation.CRR(crr_l2_reduced, test_labels)
        crr_cosine_normal_rate = PerformanceEvaluation.CRR(crr_cosine_normal, test_labels)
        crr_cosine_reduced_rate = PerformanceEvaluation.CRR(crr_cosine_reduced, test_labels)
        
        CRR_RESULTS['Number of Dimensions'].append(dimension_target)
        CRR_RESULTS['crr_l1_normal'].append(crr_l1_normal_rate)
        CRR_RESULTS['crr_l1_reduced'].append(crr_l1_reduced_rate)
        CRR_RESULTS['crr_l2_normal'].append(crr_l2_normal_rate)
        CRR_RESULTS['crr_l2_reduced'].append(crr_l2_reduced_rate)
        CRR_RESULTS['crr_cosine_normal'].append(crr_cosine_normal_rate)
        CRR_RESULTS['crr_cosine_reduced'].append(crr_cosine_reduced_rate)
        
        COSINE_SIMILARITY = cosine_similarity
        COSINE_PREDS = cosine_preds

    # Formatting the CRR results into pandas dataframe
    CRR_RESULTS = pd.DataFrame(CRR_RESULTS)
    CRR_RESULTS.set_index("Number of Dimensions", inplace=True)
    
    PerformanceEvaluation.print_CRR_tables(CRR_RESULTS)
    PerformanceEvaluation.plot_CRR_curves(CRR_RESULTS)
    
    #FNMR vs FMR
    fmr_list = []
    fnmr_list = []
    thresholds = [0.446, 0.472, 0.502, 8,10,12,15]

    for threshold in thresholds:
        fmr, fnmr = PerformanceEvaluation.false_rate(COSINE_SIMILARITY, test_labels, threshold, COSINE_PREDS)
        fmr_list.append(fmr)
        fnmr_list.append(fnmr)
    
    PerformanceEvaluation.create_fmr_fnmr_table(thresholds, fmr_list, fnmr_list)
    PerformanceEvaluation.plot_ROC(fmr_list, fnmr_list)
        
if __name__ == "__main__":    
    main()



