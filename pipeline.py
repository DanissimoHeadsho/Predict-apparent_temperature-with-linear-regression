import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from model import GDRegressor

FEATURE_COLUMNS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
TARGET_COLUMN = "apparent_temperature"

class StandardScaler: #класс для скейла данных, чтобы каждый признак вносил равный вклад
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_model(data: pd.DataFrame):
    #возвращаем обученную модель, метрики и худшие предсказания (у которых ошибка в 3 раза больше mae)
    if data.empty:
        raise ValueError("Данных нет")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, Y_train, Y_test= train_test_split(X, y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg = GDRegressor(alpha=0.3, n_iter=10000)
    reg.fit(X_train_scaled, Y_train)

    y_pred = reg.predict(X_test_scaled)
    errors = np.abs(Y_test - y_pred)
    mae = np.mean(abs(errors))
    rmse = (np.mean(errors ** 2)) ** 0.5
    r2 = 1 - np.sum(errors ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2)

    anomalies = X_test[errors >= 3 * mae].copy()
    anomalies[TARGET_COLUMN] = Y_test[errors >= 3 * mae]
    anomalies["prediction"] = y_pred[errors >= 3 * mae]
    anomalies["absolute_error"] = errors[errors >= 3 * mae]
    anomalies.sort_values(by='absolute_error', ascending=False, inplace=True)
    return reg, {"MAE": mae, "RMSE": rmse, "R2": r2}, anomalies, scaler
