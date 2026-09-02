import numpy as np
from abc import ABC, abstractmethod
from interfaces import (
    LearningRateSchedule,
    AbstractOptimizer,
    LinearRegressionInterface,
)


# ===== Learning Rate Schedules =====
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
        """
        returns: float, learning rate для iteration шага обучения
        """
        # TODO: реализовать формулу затухающего шага обучения
        new_lr = self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p
        return new_lr


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """

    def __init__(
        self,
        lr_schedule: LearningRateSchedule = TimeDecayLR(),
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        # TODO: implement
        # в конце также приcваивает атрибуту модели полученный loss_history
        X, y = self.model.X_train, self.model.y_train

        for _ in range(self.max_iter):
            # loss before current step
            self.model.loss_history.append(self.model.compute_loss(X, y))

            # update weights and get w_{k+1} - w_k
            delta_w = self._step()

            if np.any(np.isnan(delta_w)):
                break

            if np.linalg.norm(delta_w) ** 2 < self.tolerance:
                break

        # loss after the optimization
        self.model.loss_history.append(self.model.compute_loss(X, y))


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать vanilla градиентный спуск
        # Можно использовать атрибуты класса self.model
        grad = self.model.compute_gradients()

        lr = self.lr_schedule.get_lr(self.iteration)

        delta_w = -lr * grad
        self.model.w += delta_w

        return delta_w


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели
        n_samples = self.model.X_train.shape[0]

        assert n_samples >= self.batch_size
        batch_indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
        X_batch = self.model.X_train[batch_indices, :]
        y_batch = self.model.y_train[batch_indices]

        grad = self.model.compute_gradients(X_batch, y_batch)

        lr = self.lr_schedule.get_lr(self.iteration)

        delta_w = -lr * grad
        self.model.w += delta_w

        return delta_w


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать SAG
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            # TODO: инициализировать хранилища при первом вызове
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)

        assert num_objects >= self.batch_size
        batch_indices = np.random.choice(
            num_objects, size=self.batch_size, replace=False
        )

        for idx in batch_indices:
            X_i = X_train[idx : idx + 1]
            y_i = y_train[idx : idx + 1]

            new_grad = self.model.compute_gradients(X_i, y_i)

            self.grad_sum -= self.grad_memory[idx]
            self.grad_sum += new_grad

            self.grad_memory[idx] = new_grad

        avg_grad = self.grad_sum / num_objects

        lr = self.lr_schedule.get_lr(self.iteration)

        delta_w = -lr * avg_grad
        self.model.w += delta_w

        return delta_w


class MomentumDescent(BaseDescent):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать градиентный спуск с моментумом

        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)

        X_train = self.model.X_train
        y_train = self.model.y_train

        new_grad = self.model.compute_gradients(X_train, y_train)
        lr = self.lr_schedule.get_lr(self.iteration)

        new_velocity = self.beta * self.velocity + lr * new_grad

        delta_w = -new_velocity
        self.model.w += delta_w

        self.velocity = new_velocity

        return delta_w


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать Adam по формуле из ноутбука
        X_train = self.model.X_train
        y_train = self.model.y_train

        if self.m is None or self.v is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)

        t = self.iteration + 1  # +1 because Adam's uses 1-index iterations
        grad = self.model.compute_gradients(X_train, y_train)
        lr = self.lr_schedule.get_lr(self.iteration)

        new_m = self.beta1 * self.m + (1 - self.beta1) * grad

        new_v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        norm_m = new_m / (1 - self.beta1**t)
        norm_v = new_v / (1 - self.beta2**t)

        delta_w = -lr * norm_m / (np.sqrt(norm_v) + self.eps)

        self.model.w += delta_w

        self.m = new_m
        self.v = new_v

        return delta_w


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений
    """

    def __init__(self):
        self.model = None

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект
        X = self.model.X_train
        y = self.model.y_train

        self.model.w = self.model.loss_function.analytic_solution(X, y)
