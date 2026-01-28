import numpy as np
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, Optional, Union

class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()

class LinearRegression:
    def __init__(self, optimizer: Optional[Union[BaseDescent, str]] = None,
                 l2_coef: float = 0.0,
                 tolerance: float = 1e-6,
                 max_iter: int = 1000,
                 loss_function: LossFunction = LossFunction.MSE):
        self.optimizer = optimizer
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError('Model is not fitted yet. Call fit() first')
        return X @ self.w

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        
        if self.loss_function is LossFunction.MSE:
            loss = np.mean((y_pred - y) ** 2)
        elif self.loss_function is LossFunction.MAE:
            loss = np.mean(np.abs(y_pred - y))
        elif self.loss_function is LossFunction.LogCosh:
            loss = np.mean(np.log(np.cosh(y_pred - y)))
        elif self.loss_function is LossFunction.Huber:
            delta = 1.0
            diff = y_pred - y
            abs_diff = np.abs(diff)
            is_small_error = abs_diff <= delta
            squared_loss = 0.5 * diff ** 2
            linear_loss = delta * (abs_diff - 0.5 * delta)
            loss = np.mean(np.where(is_small_error, squared_loss, linear_loss))
        else:
            raise NotImplementedError(f"Loss for {self.loss_function} not implemented")
        
        if self.l2_coef > 0 and self.w is not None:
            loss += self.l2_coef * np.sum(self.w ** 2)
            
        return float(loss)

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        
        if self.loss_function is LossFunction.MSE:
            grad = (2.0 / n_samples) * (X.T @ (y_pred - y))
        elif self.loss_function is LossFunction.MAE:
            grad = (1.0 / n_samples) * (X.T @ np.sign(y_pred - y))
        elif self.loss_function is LossFunction.LogCosh:
            grad = (1.0 / n_samples) * (X.T @ np.tanh(y_pred - y))
        elif self.loss_function is LossFunction.Huber:
            delta = 1.0
            diff = y_pred - y
            abs_diff = np.abs(diff)
            is_small_error = abs_diff <= delta
            grad_small = diff
            grad_large = delta * np.sign(diff)
            grad = (1.0 / n_samples) * (X.T @ np.where(is_small_error, grad_small, grad_large))
        else:
            raise NotImplementedError(f"Gradient for {self.loss_function} not implemented")
        
        if self.l2_coef > 0 and self.w is not None:
            grad += 2 * self.l2_coef * self.w
            
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.loss_history = []
        
        if self.optimizer is None or self.optimizer == "SVD":
            X = np.asarray(X)
            y = np.asarray(y)
            
            X_offset = np.zeros(X.shape[1])
            y_offset = 0
            
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            
            threshold = np.finfo(X.dtype).eps * max(X.shape) * s[0]
            s_inv = np.zeros_like(s)
            s_inv[s > threshold] = 1 / s[s > threshold]
            
            self.w = Vt.T @ np.diag(s_inv) @ U.T @ y
            
            self.loss_history.append(self.compute_loss(X, y))
            
        elif isinstance(self.optimizer, BaseDescent):
            n_features = X.shape[1]
            self.w = np.zeros(n_features)
            
            if hasattr(self.optimizer, 'set_model'):
                self.optimizer.set_model(self)
            
            prev_w = self.w.copy()
            
            self.loss_history.append(self.compute_loss(X, y))
            
            for iteration in range(self.max_iter):
                self.optimizer.step()
                
                self.loss_history.append(self.compute_loss(X, y))
                
                weight_diff_norm = np.linalg.norm(self.w - prev_w)
                if weight_diff_norm < self.tolerance:
                    break
                
                prev_w = self.w.copy()
        else:
            n_features = X.shape[1]
            self.w = np.zeros(n_features)
            learning_rate = 0.01
            
            prev_w = self.w.copy()
            
            self.loss_history.append(self.compute_loss(X, y))
            
            for iteration in range(self.max_iter):
                grad = self.compute_gradients(X, y)
                self.w = self.w - learning_rate * grad

                self.loss_history.append(self.compute_loss(X, y))

                weight_diff_norm = np.linalg.norm(self.w - prev_w)
                if weight_diff_norm < self.tolerance:
                    break
                
                prev_w = self.w.copy()

        return self

    def get_weights(self) -> np.ndarray:
        return self.w.copy() if self.w is not None else None

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
            
        return 1 - (ss_res / ss_tot)