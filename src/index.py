import os

import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def load_images(file_list):
    '''Load images' information into arrays.'''
    images = []
    labels = []
    for file in file_list:
        img = load_img(file)
        img_array = img_to_array(img)
        images.append(img_array)
        label = int(os.path.basename(file)[:3]) # First 3 characters of the file name
        labels.append(label)
        print(label)
    
    return np.array(images), np.array(labels)


def train_validate_test_split():
    '''Determine which image files go into the train-validate-test sets.'''

    # Group images by person ID
    input_folder = 'data/images'
    images_by_person = {}
    for file_name in os.listdir(input_folder):
        person_id = int(file_name[:3]) # First 3 characters of the file name
        if person_id not in images_by_person:
            images_by_person[person_id] = []
        images_by_person[person_id].append(os.path.join(input_folder, file_name))

    # For each person, split her images into train, validation, and test sets
    # according to a ratio
    train_images = []
    validate_images = []
    test_images = []
    for person_id, images in images_by_person.items():
        np.random.seed(person_id)
        np.random.shuffle(images)
        # train:validate:test ratio is 60:20:20
        train_count = 43
        validate_count = 14
        # test_count = 15
        train_images.extend(images[:train_count])
        validate_images.extend(images[train_count: train_count + validate_count])
        test_images.extend(images[train_count + validate_count:])

    # Log the results of the split
    with open('data/logs/train_images.txt', 'w') as f:
        f.write('\n'.join(train_images))
    with open('data/logs/validate_images.txt', 'w') as f:
        f.write('\n'.join(validate_images))
    with open('data/logs/test_images.txt', 'w') as f:
        f.write('\n'.join(test_images))

    return train_images, validate_images, test_images


def build_custom_cnn():
    pass


def fit_model():
    pass


def evaluate_model():
    pass


if __name__ == '__main__':
    train_images, validate_images, test_images = train_validate_test_split()
    build_custom_cnn()
    fit_model()
    evaluate_model()
