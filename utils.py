import os
import pickle
import numpy as np
import re

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