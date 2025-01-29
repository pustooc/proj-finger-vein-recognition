import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def ensure_reproducibility():
    # Limit Tensorflow to use a single thread to ensure deterministic execution
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    # Explicitly enable determinism
    tf.config.experimental.enable_op_determinism()
    # Set seeds for Tensorflow, Numpy, and Python
    tf.keras.utils.set_random_seed(420)


def train_validate_test_split():
    '''
    For each person, allocate image files to the train-validate-test sets.
    
    Returns:
        train_df (DataFrame): The train set with the columns image_file (x) and person_id (y)
        validate_df (DataFrame): The validate set with the columns image_file (x) and person_id (y)
        test_df (DataFrame): The test set set with the columns image_file (x) and person_id (y)

    Hyperparameters:
        Type of split
        Ratio of the split
        Number of images per person
    '''

    # Group images by person ID
    INPUT_FOLDER = 'data/images'
    images_by_person = {}
    for file_name in os.listdir(INPUT_FOLDER):
        person_id = file_name[:3] # First 3 characters of the file name
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
        np.random.seed(int(person_id))
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


def load_images(train_df, validate_df, test_df):
    '''
    Create generators for the train-test-split sets to load images in batches.
    Also augment images here.

    Returns:
        train_generator (DataFrameIterator): The generator for train images
        validate_generator (DataFrameIterator): The generator for validate images
        test_generator (DataFrameIterator): The generator for test images
    
    Hyperparameters:
        Image size
        Training batch size
    '''

    train_augmentation = ImageDataGenerator(
        rescale=1.0 / 255 # Normalise pixel values from 0 - 255 to 0 - 1
    )
    train_generator = train_augmentation.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_file',
        y_col='person_id',
        color_mode='grayscale', # Ensure pixel dimension is 1
        class_mode='categorical', # One hot encode for multi-class classification
        shuffle=True, # Reordering images is required when training in new epochs
        seed=420,
        target_size=(144, 192),
        batch_size=32
    )
    validate_augmentation = ImageDataGenerator(
        rescale=1.0 / 255 # Normalise pixel values from 0 - 255 to 0 - 1
    )
    validate_generator = validate_augmentation.flow_from_dataframe(
        dataframe=validate_df,
        x_col='image_file',
        y_col='person_id',
        color_mode='grayscale', # Ensure pixel dimension is 1
        class_mode='categorical', # One hot encode for multi-class classification
        shuffle=False, # Image order can remain the same when validating in new epochs
        target_size=(144, 192),
        batch_size=32 # Set as large as memory can handle for increased evaluation speed
    )
    test_augmentation = ImageDataGenerator(
        rescale=1.0 / 255 # Normalise pixel values from 0 - 255 to 0 - 1
    )
    test_generator = test_augmentation.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_file',
        y_col='person_id',
        color_mode='grayscale', # Ensure pixel dimension is 1
        class_mode='categorical', # One hot encode for multi-class classification
        shuffle=False, # Image order can remain the same when testing in new epochs
        target_size=(144, 192),
        batch_size=32 # Set as large as memory can handle for increased evaluation speed
    )

    return train_generator, validate_generator, test_generator


def define_custom_cnn():
    '''
    Returns:
        model (Sequential): An untrained CNN

    Hyperparameters:
        Number of convolution + pooling layers
        Convolution filters, kernel size, stride, padding, and activation function
        Pooling type, size, stride, and padding
        Number of fully connected layers (FCL)
        FCL type, nodes, and activation function
        Number of dropout layers
        Dropout rates
        Output layer activation function
        Optimiser
    '''

    CLASSES_COUNT = 100

    model = models.Sequential()
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu',
        input_shape=(144, 192, 1)
    ))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    ))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(CLASSES_COUNT, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Log the resultant architecture
    model.summary()

    return model


def train_model(train_generator, validate_generator, model):
    '''
    Update the CNN with trained weights.

    Hyperparameters:
        Number of epochs
    '''

    model.fit(train_generator, epochs=12, validation_data=validate_generator)


def evaluate_model(test_generator, model):
    predictions = model.predict(test_generator)
    y_predicted = np.argmax(predictions, axis=1)
    print(accuracy_score(test_generator.classes, y_predicted))


if __name__ == '__main__':
    ensure_reproducibility()
    train_df, validate_df, test_df = train_validate_test_split()
    train_generator, validate_generator, test_generator = load_images(train_df, validate_df, test_df)
    model = define_custom_cnn()
    train_model(train_generator, validate_generator, model)
    evaluate_model(test_generator, model)
