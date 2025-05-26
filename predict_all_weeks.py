import pickle
import pandas as pd
import numpy as np
from preprocess_data import Y_COLUMN_NAME
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL = 'xgboost_sub_2000'
X_TRAIN_PATH = 'data/transformed/sub_2000/x_train.csv'
Y_TRAIN_PATH = 'data/processed/sub_2000/y_train.csv'


def predict_per_week():
    with open(f'models/{MODEL}.pkl', 'rb') as f:
        model = pickle.load(f)

    x_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)

    x_train[Y_COLUMN_NAME] = y_train[Y_COLUMN_NAME].values

    weeks = x_train[['year', 'calendar_week']].drop_duplicates().sort_values(['year', 'calendar_week'])

    weekly_results = []

    for _, row in weeks.iterrows():
        year, week = row['year'], row['calendar_week']
        x_train_for_week = x_train[(x_train['year'] == year) & (x_train['calendar_week'] == week)].copy()

        if len(x_train_for_week) == 0 or Y_COLUMN_NAME not in x_train_for_week.columns:
            continue

        x = x_train_for_week[model.feature_names_in_]
        y_true = x_train_for_week[Y_COLUMN_NAME].values
        y_pred = model.predict(x)
        y_pred = np.round(y_pred / 100) * 100

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        weekly_results.append({
            'year': year,
            'week': week,
            'samples': len(x_train_for_week),
            'mae': mae,
            'rmse': rmse
        })

    result_df = pd.DataFrame(weekly_results)

    print("\n=== Results for all weeks ===")
    print(f"⏹ Average MAE: {result_df['mae'].mean():.2f}")
    print(f"⏹ Average RMSE: {result_df['rmse'].mean():.2f}")

    file_path = f'predictions/{MODEL}_weekly.csv'
    result_df.to_csv(file_path, index=False)
    print(f"\n✅ Result saved in {file_path}")


if __name__ == '__main__':
    predict_per_week()


