import numpy as np
from abc import ABC, abstractmethod

# ===== Learning Rate Schedules =====
class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p
        # реализовали формулу


# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        self.update_weights()
        self.iteration += 1


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def update_weights(self):
        lr = self.lr_schedule.get_lr(self.iteration) # получение learning rate для текущей итерации
        gradient = self.model.compute_gradients(self.model.X_train, self.model.y_train) # вычисляем градиент на всей обучающей выборке
        delta_w = -lr * gradient # вычисляем разницу весов
        self.model.w += delta_w # обновляем веса модели

        return delta_w


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):
        lr = self.lr_schedule.get_lr(self.iteration) # получаем learning rate для текущей итерации
        X_train = self.model.X_train
        y_train = self.model.y_train
        n_samples = X_train.shape[0]
        # получили данные

        batch_indexes = np.random.randint(0, n_samples, self.batch_size) # случайно выбираем индексы для батча с повторениями

        X_batch = X_train[batch_indexes]
        y_batch = y_train[batch_indexes]
        # выбрали батч данных

        gradient = self.model.compute_gradients(X_batch, y_batch) # вычисляем градиет на батче

        delta_w = -lr * gradient # вычисляем разницу весов
        self.model.w += delta_w # обновляем веса модели

        return delta_w

class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        super().__init__(lr_schedule)
        self.grad_memory = None # хранилище индивидуальных градиентов
        self.grad_sum = None # сума всех градиентов

    def update_weights(self):
        lr = self.lr_schedule.get_lr(self.iteration) # получаем learning rate для текущей итерации

        # получаем данные
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        # инициализация при первом вызове
        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)

        j = np.random.randint(0, num_objects) # случайно выбираем индекс объекта

        X_j = X_train[j:j+1] # срез для сохранения размерности
        y_j = y_train[j:j+1]
        grad_new = self.model.compute_gradients(X_j, y_j)

        grad_old = self.grad_memory[j] # получаем старый градиент для этого объекта
        self.grad_sum = self.grad_sum - grad_old + grad_new # обновляем сумму градиентов
        self.grad_memory[j] = grad_new # обновляем хранилище градиентов
        avg_grad = self.grad_sum / num_objects # вычисляем средний градиент
        delta_w = -lr * avg_grad # вычисляем разницу весов
        self.model.w += delta_w # обновляем веса модели

        return delta_w


class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta=0.9):
        super().__init__(lr_schedule)
        self.beta = beta
        self.velocity = None

    def update_weights(self):
        lr = self.lr_schedule.get_lr(self.iteration) # получаем learning rate для текущей итерации
        gradient = self.model.compute_gradients(self.model.X_train, self.model.y_train)

        # инициализация скорости при первом вызове
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        
        self.velocity = self.beta * self.velocity + lr * gradient # обновление скорости
        delta_w = -self.velocity # вычисляем разницу весов
        self.model.w += delta_w # обновляем веса мдели

        return delta_w




class Adam(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr_schedule)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update_weights(self):
        lr = self.lr_schedule.get_lr(self.iteration) # получаем learning rate для текущей итерации
        gradient = self.model.compute_gradients(self.model.X_train, self.model.y_train)

        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient # обновление первого момента (среднее)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2) # обновление второго момента (дисперсия)

        # коррекция смещения
        t = self.iteration + 1
        m_hat = self.m / (1 - self.beta1 ** t)
        v_hat = self.v / (1 - self.beta2 ** t)

        delta_w = -lr * m_hat / (np.sqrt(v_hat) + self.eps) # разница весов
        self.model.w += delta_w

        return delta_w