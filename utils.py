import os
import pickle
import numpy as np
import re
import cv2

def save_features(features, labels, filepath):
    """Save features and labels to a file."""
    with open(filepath, 'wb') as file:
        pickle.dump((features, labels), file)

def load_features(filepath):
    """Load features and labels from a file."""
    with open(filepath, 'rb') as file:
        return pickle.load(file)
    
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