import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_validate_test_split():
    '''
    Allocate image files to the train-validate-test sets for each person.
    Returns: three dataframes representing the train-validate-test sets, each with an
    image_file column (x) and a person_id column (y).
    Hyperparameters: ratio of the split; number of images per person.
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
    Hyperparameters: batch size.
    '''

    BATCH_SIZE = 32

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
        batch_size=BATCH_SIZE
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
        batch_size=BATCH_SIZE
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
        batch_size=BATCH_SIZE
    )

    return train_generator, validate_generator, test_generator


def define_custom_cnn():
    '''
    Define the architecture of a custom CNN.
    Returns: an untrained CNN.
    Hyperparameters: number of convolution + pooling layers; convolution filters,
    kernel size, stride, padding, and activation function; pooling type, size,
    stride, and padding; number of fully connected layers (FCL); FCL nodes and
    activation function; number of dropout layers; dropout rates; output layer
    activation function; optimiser.
    '''

    CLASSES_COUNT = 100

    model = Sequential()
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        input_shape=(576, 768, 1)
    ))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same'
    ))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(CLASSES_COUNT, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model


def train_model(train_generator, validate_generator, model):
    '''Hyperparameters: number of epochs.'''

    return model.fit(train_generator, epochs=20, validation_data=validate_generator)


def predict_test_classes(test_generator, model):
    predictions = model.predict(test_generator)
    y_predicted = np.argmax(predictions, axis=1)
    print(classification_report(test_generator.classes, y_predicted))


if __name__ == '__main__':
    train_df, validate_df, test_df = train_validate_test_split()
    train_generator, validate_generator, test_generator = load_images(train_df, validate_df, test_df)
    model = define_custom_cnn()
    history = train_model(train_generator, validate_generator, model)
    predict_test_classes(test_generator, model)
