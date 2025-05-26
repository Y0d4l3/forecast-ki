import numpy as np
import pandas as pd

FEATURES_TO_USE = []
Y_COLUMN_NAME = 'production'


def features_to_use(df):
    if len(FEATURES_TO_USE) != 0:
        return FEATURES_TO_USE
    else:
        return df.columns.difference([Y_COLUMN_NAME]).tolist()


def preprocess_df(df):
    df.drop('gmc', axis=1, inplace=True)
    df.drop('parameter_id', axis=1, inplace=True)
    df.drop('ready', axis=1, inplace=True)

    df.loc[:, 'week_sin'] = np.sin(2 * np.pi * df['calendar_week'] / 52)
    df.loc[:, 'week_cos'] = np.cos(2 * np.pi * df['calendar_week'] / 52)

    boolean_columns = ['produce_with_blank', 'blank']
    for column in boolean_columns:
        df[column] = df[column].astype(int)

    return df


def time_split_by_week(df, test_ratio=0.2):
    df['year_week'] = df['year'].astype(str) + '-' + df['calendar_week'].astype(str).str.zfill(2)

    unique_weeks = df['year_week'].drop_duplicates().tolist()

    n_total_weeks = len(unique_weeks)
    n_test_weeks = int(n_total_weeks * test_ratio)

    test_weeks = set(unique_weeks[-n_test_weeks:])

    test_df = df[df['year_week'].isin(test_weeks)].copy()
    train_df = df[~df['year_week'].isin(test_weeks)].copy()

    train_df.drop(columns='year_week', inplace=True)
    test_df.drop(columns='year_week', inplace=True)

    x_train = train_df[features_to_use(train_df)]
    x_test = test_df[features_to_use(test_df)]

    y_train = train_df[Y_COLUMN_NAME]
    y_test = test_df[Y_COLUMN_NAME]

    return x_train, x_test, y_train, y_test


def main():
    #df = pd.read_csv('data/raw.csv')
    df = pd.read_csv('data/test_2025_13.csv')

    df = df[(df['production'] <= 2000)].copy()

    df = df.sort_values(by=['year', 'calendar_week']).reset_index(drop=True)

    preprocessed_df = preprocess_df(df)

    #x_train, x_test, y_train, y_test = time_split_by_week(preprocessed_df)

    #x_train.to_csv('data/processed/sub_2000/x_train.csv', index=False)
    #x_test.to_csv('data/processed/sub_2000/x_test.csv', index=False)
    #y_train.to_csv('data/processed/sub_2000/y_train.csv', index=False)
    #y_test.to_csv('data/processed/sub_2000/y_test.csv', index=False)

    preprocessed_df.to_csv('data/processed/test_2025_13.csv', index=False)


if __name__ == '__main__':
    main()
