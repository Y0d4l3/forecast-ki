import pickle
import pandas as pd
import numpy as np
from train_model import Y_COLUMN_NAME


def predict_for_week(year, week):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('data/transformed_data.csv')
    #df = data[(data['production'] <= 2000)]

    df = df[(df['year'] == year) & (df['calendar_week'] == week)].copy()
    if len(df) == 0:
        print(f'No data found for year={year}, week={week}.')
        return None

    actual_production_values = df[Y_COLUMN_NAME].values if Y_COLUMN_NAME in df.columns else None
    ids = df['id'].values if 'id' in df.columns else None

    x_predict = df[model.feature_names_in_]

    y_predict = model.predict(x_predict)

    y_predict = np.round(y_predict / 100) * 100

    result_df = pd.DataFrame({
        'id': ids,
        'predicted_production_values': y_predict.astype(int)
    })

    if actual_production_values is not None:
        result_df['actual_production_values'] = actual_production_values

    return result_df


if __name__ == '__main__':
    result = predict_for_week(year=2025, week=12)
    if result is not None:
        file_path = 'data/prediction.csv'
        result.to_csv(file_path, index=False)
        print(f'prediction stored in {file_path}.')
        mae = (result['actual_production_values'] - result['predicted_production_values']).abs().mean()
        rmse = np.sqrt(((result['actual_production_values'] - result['predicted_production_values']) ** 2).mean())
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
