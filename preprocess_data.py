import numpy as np
import pandas as pd


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


def main():
    df = pd.read_csv('data/raw_data.csv')

    preprocessed_df = preprocess_df(df)
    preprocessed_df.to_csv('processed_data.csv', index=False)


if __name__ == '__main__':
    main()
