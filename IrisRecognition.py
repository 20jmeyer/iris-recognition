import os
import cv2
import IrisLocalization
import IrisNormalization
from IrisNormalization import normalize_iris


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


def main():
    base_folder = './database'  # Need to update to the correct path
    images = load_images_from_folder(base_folder)

    # Access training and testing images
    train_images = images['train']
    test_images = images['test']

    create_output_dir("localized_output", "train")
    create_output_dir("localized_output", "test")
    create_output_dir("norm_output", "train")
    create_output_dir("norm_output", "test")
        
    for image in train_images:
        # Localization
        iris, _ = IrisLocalization.locate_iris(image)
        save_name =  './localized_output/train/'+os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(save_name, iris)

        # Normalization
        norm_iris = IrisNormalization.normalize_iris(iris)
        norm_name = './norm_output/train/' + os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(norm_name, norm_iris)
        
    for image in test_images:
        # Localization
        iris, _ = IrisLocalization.locate_iris(image)
        save_name =  './localized_output/test/'+os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(save_name, iris)

        # Normalization
        norm_iris = IrisNormalization.normalize_iris(iris)
        norm_name = './norm_output/test/' + os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(norm_name, norm_iris)

main()


