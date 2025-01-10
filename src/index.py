import os

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def train_validate_test_split():
    '''
    Determine which image files go into the train-validate-test sets.
    Returns: three dataframes representing the train-validate-test split, each with an
    image_file column (x) and a person_id column (y).
    Hyperparameters: ratio of the split; number of images per person.
    '''

    # Group images by person ID
    INPUT_FOLDER = 'data/images'
    images_by_person = {}
    for file_name in os.listdir(INPUT_FOLDER):
        person_id = int(file_name[:3]) # First 3 characters of the file name
        if person_id not in images_by_person:
            images_by_person[person_id] = []
        images_by_person[person_id].append(os.path.join(INPUT_FOLDER, file_name))

    # For each person, split her images into train-validate-test sets
    # according to a ratio
    train_images = []
    train_labels = []
    validate_images = []
    validate_labels = []
    test_images = []
    test_labels = []
    for person_id, images in images_by_person.items():
        # Randomise the order of image selection with a seed for reproducibility
        np.random.seed(person_id)
        np.random.shuffle(images)
        # 60:20:20 is the train:validate:test ratio
        # 72 images per person
        TRAIN_COUNT = 43
        VALIDATE_COUNT = 14
        TEST_COUNT = 15
        train_images.extend(images[:TRAIN_COUNT])
        train_labels.extend([person_id] * TRAIN_COUNT)
        validate_images.extend(images[TRAIN_COUNT: TRAIN_COUNT + VALIDATE_COUNT])
        validate_labels.extend([person_id] * VALIDATE_COUNT)
        test_images.extend(images[TRAIN_COUNT + VALIDATE_COUNT:])
        test_labels.extend([person_id] * TEST_COUNT)
    # Initialise the sets as dataframes with labels so that
    # flow_with_dataframe can be used
    train_df = pd.DataFrame({
        'image_file': train_images,
        'person_id': train_labels
    })
    validate_df = pd.DataFrame({
        'image_file': validate_images,
        'person_id': validate_labels
    })
    test_df = pd.DataFrame({
        'image_file': test_images,
        'person_id': test_labels
    })

    # Log the results of the split
    train_df.to_csv('data/logs/train_df.csv', index=False)
    validate_df.to_csv('data/logs/validate_df.csv', index=False)
    test_df.to_csv('data/logs/test_df.csv', index=False)

    return train_df, validate_df, test_df


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
    train_df, validate_df, test_df = train_validate_test_split()
    load_images()
    define_custom_cnn()
    train_model()
    predict_test_classes()
    evaluate_model()
