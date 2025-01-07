import os

import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def train_validate_test_split():
    '''
    Determine which image files go into the train-validate-test sets.
    Hyperparameters: ratio of the split.
    '''

    # Group images by person ID
    INPUT_FOLDER = 'data/images'
    images_by_person = {}
    for file_name in os.listdir(INPUT_FOLDER):
        person_id = int(file_name[:3]) # First 3 characters of the file name
        if person_id not in images_by_person:
            images_by_person[person_id] = []
        images_by_person[person_id].append(os.path.join(INPUT_FOLDER, file_name))

    # For each person, split her images into train, validation, and test sets
    # according to a ratio
    train_images = []
    validate_images = []
    test_images = []
    for person_id, images in images_by_person.items():
        np.random.seed(person_id)
        np.random.shuffle(images)
        # train:validate:test ratio is 60:20:20
        TRAIN_COUNT = 43
        VALIDATE_COUNT = 14
        # TEST_COUNT = 15
        train_images.extend(images[:TRAIN_COUNT])
        validate_images.extend(images[TRAIN_COUNT: TRAIN_COUNT + VALIDATE_COUNT])
        test_images.extend(images[TRAIN_COUNT + VALIDATE_COUNT:])

    # Log the results of the split
    with open('data/logs/train_images.txt', 'w') as f:
        f.write('\n'.join(train_images))
    with open('data/logs/validate_images.txt', 'w') as f:
        f.write('\n'.join(validate_images))
    with open('data/logs/test_images.txt', 'w') as f:
        f.write('\n'.join(test_images))

    return train_images, validate_images, test_images


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


def define_custom_cnn():
    pass


def train_model():
    pass


def predict_test_classes():
    pass


def evaluate_model():
    pass


if __name__ == '__main__':
    train_images, validate_images, test_images = train_validate_test_split()
    load_images()
    define_custom_cnn()
    train_model()
    predict_test_classes()
    evaluate_model()
