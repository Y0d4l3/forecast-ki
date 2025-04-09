import logging
import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna

MODEL_NAME = 'xgboost'
Y_COLUMN_NAME = 'production'
FEATURES_TO_USE = []
FEATURES_TO_TRANSFORM = ['stock', 'net_raw_demand', 'preview_sum']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('model_training.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_features(df):
    if len(FEATURES_TO_USE) != 0:
        return FEATURES_TO_USE
    else:
        return [col for col in df.columns if col != Y_COLUMN_NAME]


def objective(trial, x, y):
    if MODEL_NAME == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        model = XGBRegressor(objective='reg:squarederror', **params, random_state=42)
    elif MODEL_NAME == 'random_forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 10, 50, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    elif MODEL_NAME == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10)
        }
        model = LGBMRegressor(**params, random_state=42)
    else:
        raise ValueError('Model name unknown. Use xgboost, random_forest or lightgbm.')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_index, val_index in kf.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


def optimize_model(x, y, n_trials):
    def objective_wrapper(trial):
        return objective(trial, x, y)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_wrapper, n_trials=n_trials)
    logger.info(f'Best Hyperparameters: {study.best_params}')
    return study.best_params


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f'Evaluation -> RMSE: {rmse:.4f}, MAE: {mae:.4f}')
    return rmse, mae


def save_artifacts(model):
    with open('model.pkl', 'wb') as f:
        # noinspection PyTypeChecker
        pickle.dump(model, f)


def main():
    df = pd.read_csv('data/transformed_data.csv')
    #df = data[(data['production'] <= 2000)]

    training_df = df[get_features(df)]

    x = training_df[training_df.columns.difference([Y_COLUMN_NAME]).tolist()]
    y = df[Y_COLUMN_NAME]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    logger.info('Training started with: ' + MODEL_NAME)
    best_params = optimize_model(x_train.to_numpy(), y_train.to_numpy(), n_trials=10)

    if MODEL_NAME == 'xgboost':
        model = XGBRegressor(objective='reg:squarederror', **best_params, random_state=42)
    elif MODEL_NAME == 'random_forest':
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    elif MODEL_NAME == 'lightgbm':
        model = LGBMRegressor(**best_params, random_state=42)
    else:
        raise ValueError('Model name unknown. Use xgboost, random_forest or lightgbm.')

    model.fit(x_train, y_train)

    evaluate_model(model, x_test, y_test)
    save_artifacts(model)


if __name__ == '__main__':
    main()

