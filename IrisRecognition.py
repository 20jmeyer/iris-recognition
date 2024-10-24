import os
import cv2
import IrisLocalization

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

def main():
    base_folder = './database'  # Need to update to the correct path
    images = load_images_from_folder(base_folder)

    # Access training and testing images
    train_images = images['train']
    test_images = images['test']


    output_path = './output/train'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = './output/test'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for image in train_images:
        iris, _ = IrisLocalization.locate_iris(image)
        save_name =  './output/train/'+os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(save_name, iris)
        
    for image in test_images:
        iris, _ = IrisLocalization.locate_iris(image)
        save_name =  './output/test/'+os.path.basename(image)[:-4] + '_iris.bmp'
        cv2.imwrite(save_name, iris)
main()


