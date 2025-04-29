import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, StandardScaler

from train_model import get_features

FEATURES_TO_TRANSFORM = ['stock', 'net_raw_demand', 'preview_sum', 'production_demand']


def create_feature_engineering_pipeline(features_to_transform):
    log_pipeline = Pipeline([
        ('log', FunctionTransformer(np.log1p, validate=False))
    ])

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


def transform_features(df):
    if not all(FEATURE in get_features(df) for FEATURE in FEATURES_TO_TRANSFORM):
        raise ValueError('Not all features to transform are present in features to use.')

    preprocessor = create_feature_engineering_pipeline(FEATURES_TO_TRANSFORM)

    transformed_array = preprocessor.fit_transform(df[FEATURES_TO_TRANSFORM])

    transformed_df = pd.DataFrame(transformed_array, columns=FEATURES_TO_TRANSFORM)

    remaining_df = df.drop(columns=FEATURES_TO_TRANSFORM)

    final_df = pd.concat([transformed_df, remaining_df], axis=1)

    return final_df


def main():
    df = pd.read_csv('data/processed_data.csv')

    mask_under_2000 = df['production_demand'] <= 2000
    transformed_df = transform_features(df)
    df = transformed_df[mask_under_2000].reset_index(drop=True)

    file_path = 'data/transformed_data.csv'
    df.to_csv(file_path, index=False)
    print(f'transformed data stored in {file_path}.')


if __name__ == '__main__':
    main()
