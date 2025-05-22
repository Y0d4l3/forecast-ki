import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from train_model import get_features

SUB_2000 = False
Y_COLUMN_NAME = 'production'
FOLDER_PATH = 'data/processed/'


def preprocess_df(df):
    if SUB_2000:
        df = df[(df['production_demand'] <= 2000)]

    df.drop('gmc', axis=1, inplace=True)
    df.drop('parameter_id', axis=1, inplace=True)
    df.drop('ready', axis=1, inplace=True)

    df.loc[:, 'week_sin'] = np.sin(2 * np.pi * df['calendar_week'] / 52)
    df.loc[:, 'week_cos'] = np.cos(2 * np.pi * df['calendar_week'] / 52)

    boolean_columns = ['produce_with_blank', 'blank']
    for column in boolean_columns:
        df[column] = df[column].astype(int)

    return df


def split_df(df):
    df = df.sort_values(by=['year', 'calendar_week'])

    training_df = df[get_features(df)]

    x = training_df[training_df.columns.difference([Y_COLUMN_NAME]).tolist()]
    y = df[Y_COLUMN_NAME]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train.to_csv(f'{FOLDER_PATH}x_train.csv', index=False)
    x_test.to_csv(f'{FOLDER_PATH}x_test.csv', index=False)
    y_train.to_csv(f'{FOLDER_PATH}y_train.csv', index=False)
    y_test.to_csv(f'{FOLDER_PATH}y_test.csv', index=False)


def main():
    df = pd.read_csv('data/raw.csv')
    preprocessed_df = preprocess_df(df)
    split_df(preprocessed_df)

    #split_df(preprocessed_df)
    print(f'data stored in {FOLDER_PATH}.')


if __name__ == '__main__':
    main()
