# 📈 Forecast-KI – Time Series Forecasting with XGBoost

This project demonstrates a machine learning approach to **time series forecasting** using **XGBoost**, with additional comparisons to **Random Forest** and **LightGBM**. It includes Jupyter notebooks for experimentation and model evaluation.

## 🚀 Overview

The goal of this project is to build a reliable forecasting model that predicts future values of a given time series (e.g. sales or production volumes) based on historical data and engineered features.

- 🔧 Core model: [XGBoost](https://xgboost.readthedocs.io/)
- 📊 Baseline comparisons: RandomForest, LightGBM
- 🧪 Experimentation via Jupyter Notebooks
- 📁 Modular code structure for easy extension
- 🧠 Focus on practical performance and interpretability

## 🧠 Features

- Feature engineering for time-based data
- Hyperparameter tuning using [Optuna](https://optuna.org/)
- Cross-validation using `TimeSeriesSplit`
- Metrics: MAE, RMSE, MAPE
- Visualizations for predictions vs. actual values
- Model export for deployment
