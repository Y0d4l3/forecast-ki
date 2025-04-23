import pickle
import pandas as pd
import numpy as np
from train_model import Y_COLUMN_NAME
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL = 'xgboost'
DATA_PATH = 'data/transformed_data.csv'

def predict_per_week():
    with open(f'models/{MODEL}.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(DATA_PATH)
    weeks = df[['year', 'calendar_week']].drop_duplicates().sort_values(['year', 'calendar_week'])

    weekly_results = []

    for _, row in weeks.iterrows():
        year, week = row['year'], row['calendar_week']
        week_df = df[(df['year'] == year) & (df['calendar_week'] == week)].copy()

        if len(week_df) == 0 or Y_COLUMN_NAME not in week_df.columns:
            continue

        x = week_df[model.feature_names_in_]
        y_true = week_df[Y_COLUMN_NAME].values
        y_pred = model.predict(x)
        y_pred = np.round(y_pred / 100) * 100

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        weekly_results.append({
            'year': year,
            'week': week,
            'samples': len(week_df),
            'mae': mae,
            'rmse': rmse
        })

        print(f"Year {year}, Week {week} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, Samples: {len(week_df)}")

    result_df = pd.DataFrame(weekly_results)

    print("\n=== Results for all weeks ===")
    print(f"⏹ Average MAE: {result_df['mae'].mean():.2f}")
    print(f"⏹ Average RMSE: {result_df['rmse'].mean():.2f}")

    file_path = f'prediction/{MODEL}_weekly.csv'
    result_df.to_csv(file_path, index=False)
    print(f"\n✅ Result saved in {file_path}")


if __name__ == '__main__':
    predict_per_week()


