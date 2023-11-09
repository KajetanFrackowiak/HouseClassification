from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder
from pandas import DataFrame

from datasets import Datasets


def one_hot_encode(data: Datasets) -> Datasets:
    columns = ['Zip_area', 'Zip_loc', 'Room']
    encoder = OneHotEncoder(drop='first').fit(data.X_train[columns])
    data.X_train_encoded = transform_and_replace_one_hot(columns, data.X_train, encoder)
    data.X_test_encoded = transform_and_replace_one_hot(columns, data.X_test, encoder)
    return data


def transform_and_replace_one_hot(columns: list[str], df: DataFrame, encoder: OneHotEncoder):
    transformed = DataFrame(encoder.transform(df[columns]).toarray(), index=df.index).add_prefix('enc')
    return df.drop(columns, axis=1).join(transformed)


def ordinal_encode(data: Datasets) -> Datasets:
    columns = ['Zip_area', 'Zip_loc', 'Room']
    encoder = OrdinalEncoder().fit(data.X_train[columns])
    data.X_train_encoded = transform_and_replace_ordinal(columns, data.X_train, encoder)
    data.X_test_encoded = transform_and_replace_ordinal(columns, data.X_test, encoder)
    return data


def transform_and_replace_ordinal(columns: list[str], df: DataFrame, encoder: OrdinalEncoder):
    transformed = DataFrame(encoder.transform(df[columns]), index=df.index).add_suffix('_ord')
    return df.drop(columns, axis=1).join(transformed)


def target_encoder(data: Datasets) -> Datasets:
    columns = ['Room', 'Zip_area', 'Zip_loc']
    encoder = TargetEncoder(cols=columns)
    data.X_train_encoded = encoder.fit_transform(data.X_train, data.y_train)
    data.X_test_encoded = encoder.transform(data.X_test)
    return data


def transform_and_replace_target(columns: list[str], df: DataFrame, encoder: TargetEncoder):
    transformed = encoder.transform(df[columns])
    transformed.columns = [f"{col}_encoded" for col in columns]
    return df.join(transformed, rsuffix='_encoded')


