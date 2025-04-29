import pickle
import pandas as pd
import numpy as np
from train_model import Y_COLUMN_NAME

MODEL = 'xgboost_2000'
DATA_PATH = 'data/transformed_data.csv'
YEAR = 2025
WEEK = 5

def predict_for_week(year, week):
    with open(f'models/{MODEL}.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(DATA_PATH)

    df = df[(df['year'] == year) & (df['calendar_week'] == week)].copy()
    if len(df) == 0:
        print(f'No data found for year={year}, week={week}')
        return None

    actual = df[Y_COLUMN_NAME].values if Y_COLUMN_NAME in df.columns else None
    ids = df['id'].values if 'id' in df.columns else None

    x_predict = df[model.feature_names_in_]

    y_predict = model.predict(x_predict)

    y_predict = np.round(y_predict / 100) * 100

    result_df = pd.DataFrame({
        'id': ids,
        'predicted': y_predict.astype(int)
    })

    if actual is not None:
        result_df['actual'] = actual

    return result_df


if __name__ == '__main__':
    result_df = predict_for_week(year=YEAR, week=WEEK)
    if result_df is not None:
        mae = (result_df['actual'] - result_df['predicted']).abs().mean()
        rmse = np.sqrt(((result_df['actual'] - result_df['predicted']) ** 2).mean())

        print(f"\n=== Results for {YEAR}-{WEEK} ===")
        print(f"⏹ Average MAE: {mae:.2f}")
        print(f"⏹ Average RMSE: {rmse:.2f}")

        file_path = f'predictions/{MODEL}_{YEAR}_{WEEK}.csv'
        result_df.to_csv(file_path, index=False)
        print(f"\n✅ Result saved in {file_path}")
