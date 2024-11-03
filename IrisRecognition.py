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
        
        save_features(train_features, train_labels, train_features_path)
        save_features(test_features, test_labels, test_features_path)

        # Fitting the dimension reduction model
        train_features = np.array(train_features)


        crr_l1 = []
        crr_l2 = []
        crr_cosine = []
        

        print("test labels", test_labels)
        dimensions = [30, 60, 80, 100, 108] #Different number of dimensions to try
        cosine_similarity = None
        cosine_preds = None
        for dimension_target in dimensions:
            reduced_model, class_centers = IrisMatching.reduce_dimensionality(
                train_features, train_labels, n_components=dimension_target
            )
            # Match Iris for the 3 distance measures
            test_features = np.array(test_features)
            for i, feature in enumerate(test_features):

                L1_test_features, L1_test_class, L1_test_similarity, L1_test_features_reduced, \
                L1_test_class_reduced, L1_test_similarity_reduced = IrisMatching.match_iris(
                    feature, class_centers, reduced_class_centers,reduced_model, distance_type='L1'
                )
                crr_l1.append(PerformanceEvaluation.CRR(L1_test_class_reduced,test_labels))
                L2_test_features, L2_test_class, L2_test_similarity, L2_test_features_reduced, \
                L2_test_class_reduced, L2_test_similarity_reduced = IrisMatching.match_iris(
                    feature, class_centers, reduced_class_centers, reduced_model, distance_type='L2'
                )
                crr_l2.append(PerformanceEvaluation.CRR(L2_test_class_reduced,test_labels))
                
                
                cosine_test_features, cosine_test_class, cosine_test_similarity, cosine_test_features_reduced, \
                cosine_test_class_reduced, cosine_test_similarity_reduced = IrisMatching.match_iris(
                    feature, class_centers, reduced_class_centers, reduced_model, distance_type='cosine'
                )
                crr_cosine.append(PerformanceEvaluation.CRR(cosine_test_class_reduced,test_labels))
                #Extracting out the cosine results for non-reduced dimension predictions
                cosine_similarity = cosine_test_similarity
                cosine_preds = cosine_test_class
        ##CRR    
        crr_results = pd.DataFrame({'dimensions': dimensions, 'crr_l1': crr_l1, 'crr_l2': crr_l2, 'crr_cosine': crr_cosine})
        PerformanceEvaluation.plot_CRR_curve(crr_results)
        
        #FNMR vs FMR
        thresholds = [0.446, 0.472, 0.502]
        fmr_list = []
        fnmr_list = []
        for threshold in thresholds:
            fmr, fnmr = PerformanceEvaluation.false_rate(cosine_similarity, test_labels, threshold,cosine_preds)
            fmr_list.append(fmr)
            fnmr_list.append(fnmr)
            
        PerformanceEvaluation.plot_ROC(fmr, fnmr)
        
        
if __name__ == "__main__":    
    main()


