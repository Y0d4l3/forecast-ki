from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from train_model import get_features

FEATURES_TO_TRANSFORM = ['stock', 'net_raw_demand', 'preview_sum', 'production_demand']
PROCESSED_DATA_FOLDER_PATH = 'data/processed/'
TRANSFORMED_DATA_FOLDER_PATH = 'data/transformed/'


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
    folder_path = Path(PROCESSED_DATA_FOLDER_PATH)
    for file in [f for f in folder_path.glob('*.csv') if 'x' in f.name]:
        df = pd.read_csv(file)
        transformed_df = transform_features(df)
        transformed_df.to_csv(f'{TRANSFORMED_DATA_FOLDER_PATH}{file.name}', index=False)

    #file_name = 'test_data_2025_13.csv'
    #df = pd.read_csv(f'{PROCESSED_DATA_FOLDER_PATH}{file_name}')
    #transformed_df = transform_features(df)
    #transformed_df.to_csv(f'{TRANSFORMED_DATA_FOLDER_PATH}{file_name}', index=False)

    print(f'data stored in {TRANSFORMED_DATA_FOLDER_PATH}.')




if __name__ == '__main__':
    main()
