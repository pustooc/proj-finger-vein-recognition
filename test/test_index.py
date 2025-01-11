from src.index import (
    train_validate_test_split,
    load_images
)


def test_train_validate_test_split_returns_correct_dataframes():
    train_df, validate_df, test_df = train_validate_test_split()
    train_columns = list(train_df.columns)
    validate_columns = list(validate_df.columns)
    test_columns = list(test_df.columns)

    assert train_columns[0] == 'image_file'
    assert train_columns[1] == 'person_id'
    assert validate_columns[0] == 'image_file'
    assert validate_columns[1] == 'person_id'
    assert test_columns[0] == 'image_file'
    assert test_columns[1] == 'person_id'
    # Number of rows
    assert train_df.shape[0] == 4300
    assert validate_df.shape[0] == 1400
    assert test_df.shape[0] == 1500
