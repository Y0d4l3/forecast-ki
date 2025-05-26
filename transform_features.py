import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from preprocess_data import features_to_use

FEATURES_TO_TRANSFORM = ['stock', 'net_raw_demand', 'preview_sum', 'production_demand']


def create_feature_engineering_pipeline(features_to_transform):
    quantile_pipeline = Pipeline([
        ('quantile', QuantileTransformer(output_distribution='normal', random_state=42))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', quantile_pipeline, features_to_transform)
        ],
        remainder='passthrough'
    )

    return preprocessor


def transform_features(x_train, x_test):
    if not all(FEATURE in features_to_use(x_train) for FEATURE in FEATURES_TO_TRANSFORM):
        raise ValueError('Not all features to transform are present in features to use.')

    transformer_path = 'models/transformer.pkl'
    if os.path.exists(transformer_path):
        with open(transformer_path, 'rb') as f:
            preprocessor = pickle.load(f)
    else:
        preprocessor = create_feature_engineering_pipeline(FEATURES_TO_TRANSFORM)

    transformed_array = preprocessor.fit_transform(x_train[FEATURES_TO_TRANSFORM])
    transformed_df = pd.DataFrame(transformed_array, columns=FEATURES_TO_TRANSFORM, index=x_train.index)
    remaining_df = x_train.drop(columns=FEATURES_TO_TRANSFORM)
    x_train_transformed = pd.concat([transformed_df, remaining_df], axis=1)

    transformed_test_array = preprocessor.transform(x_test[FEATURES_TO_TRANSFORM])
    transformed_test_df = pd.DataFrame(transformed_test_array, columns=FEATURES_TO_TRANSFORM, index=x_test.index)
    remaining_test_df = x_test.drop(columns=FEATURES_TO_TRANSFORM)
    x_test_transformed = pd.concat([transformed_test_df, remaining_test_df], axis=1)

    with open(transformer_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    return x_train_transformed, x_test_transformed


def transform_test():
    test = pd.read_csv('data/processed/test_2025_13.csv')
    transformer_path = 'models/transformer.pkl'
    with open(transformer_path, 'rb') as f:
        preprocessor = pickle.load(f)

    transformed_test_array = preprocessor.transform(test[FEATURES_TO_TRANSFORM])
    transformed_test_df = pd.DataFrame(transformed_test_array, columns=FEATURES_TO_TRANSFORM, index=test.index)
    remaining_test_df = test.drop(columns=FEATURES_TO_TRANSFORM)
    test_transformed = pd.concat([transformed_test_df, remaining_test_df], axis=1)

    test_transformed.to_csv('data/transformed/test_2025_13.csv', index=False)


def main():
    x_train = pd.read_csv('data/processed/x_train.csv')
    x_test = pd.read_csv('data/processed/x_test.csv')

    x_train_transformed, x_test_transformed = transform_features(x_train, x_test)

    x_train_transformed.to_csv('data/transformed/x_train.csv', index=False)
    x_test_transformed.to_csv('data/transformed/x_test.csv', index=False)


if __name__ == '__main__':
    main()
    #transform_test()

