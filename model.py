import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class GDRegressor:

    def __init__(self, alpha: float = 0.001, n_iter: int = 100, progress: bool = True) -> None:
        self.alpha: float = alpha
        self.n_iter: int = n_iter
        self.loss_history: list[float] | None = None
        self.theta_history: list[list[float]] | None = None
        self.coef_: FloatArray | None = None
        self.intercept_: float | None = None

    def fit(self, X_train: FloatArray, y_train: FloatArray) -> None:
        m, p = X_train.shape
        X_train = np.hstack((np.array(X_train), np.ones((m, 1))))  #добавляем столбец единиц для нахождения b
        y_train = np.array(y_train)
        np.random.seed(42)
        self.coef_ = np.random.random(p + 1)  # генерируем веса

        # self.theta_history = [[] for _ in range(p + 1)]
        # self.loss_history = []

        for _ in range(self.n_iter):
            h_x_i: FloatArray = X_train @ self.coef_
            errors: FloatArray = h_x_i - y_train
            self.coef_ -= (self.alpha / m) * (X_train.T @ errors) #обновляем коэф-ты и транспонируем, чтобы в i-ой строке лежали все значения i-го признака
            # self.loss_history.append(float(np.mean(errors ** 2)))
            # for j in range(p + 1):
            #     self.theta_history[j].append(self.coef_[j])
        self.intercept_ = float(self.coef_[-1])
        self.coef_ = self.coef_[0:-1]

    def predict(self, X_test: FloatArray) -> FloatArray:
        return X_test @ self.coef_ + self.intercept_