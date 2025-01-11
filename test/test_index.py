from src.index import (
    train_validate_test_split,
    load_images
)


def test_train_validate_test_split_returns_correct_dataframes():
    train_df, validate_df, test_df = train_validate_test_split()
    train_columns = list(train_df.columns)
    validate_columns = list(validate_df.columns)
    test_columns = list(test_df.columns)

    # Check column headers
    assert train_columns[0] == 'image_file'
    assert train_columns[1] == 'person_id'
    assert validate_columns[0] == 'image_file'
    assert validate_columns[1] == 'person_id'
    assert test_columns[0] == 'image_file'
    assert test_columns[1] == 'person_id'
    # Check number of rows
    assert train_df.shape[0] == 4300
    assert validate_df.shape[0] == 1400
    assert test_df.shape[0] == 1500


def test_load_images_returns_correct_generators():
    train_df, validate_df, test_df = train_validate_test_split()
    train_generator, validate_generator, test_generator = load_images(train_df, validate_df, test_df)

    # Check that one hot encoding worked
    assert len(train_generator.class_indices) == 100
    assert len(validate_generator.class_indices) == 100
    assert len(test_generator.class_indices) == 100
