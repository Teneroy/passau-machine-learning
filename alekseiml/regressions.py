from base import Regression
import numpy as np


class RandomNormalEstimator(Regression):
    def __init__(self):
        self._mean = 0
        self._std = 1

    def learn(self, features, targets):
        self._check_learn_shapes(features, targets)
        self._mean = np.mean(targets)
        self._std = np.std(targets)

    def infer(self, features):
        return np.random.normal(self._mean, self._std, (features.shape[0],))


class RandomUniformEstimator(Regression):
    def __init__(self):
        self._tmin = 0
        self._tmax = 1

    def learn(self, features, targets):
        self._check_learn_shapes(features, targets)
        self._tmin = min(self._tmin, np.min(targets))
        self._tmax = max(self._tmax, np.max(targets))

    def infer(self, features):
        return np.random.uniform(self._tmin, self._tmax, (features.shape[0],))


class LinearRegressionLMS(Regression):
    def __init__(self) -> None:
        self.w = None

    def _add_constant(self, features: np.ndarray) -> np.ndarray:
        return np.hstack((features, np.ones((len(features), 1))))

    def learn(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._check_learn_shapes(features, targets)
        features = self._add_constant(features)
        self.w = np.linalg.inv(features.T @ features) @ features.T @ targets  # there is also np.linalg.pinv

    def infer(self, features: np.ndarray) -> np.ndarray:
        self._check_infer_shapes(features)
        features = self._add_constant(features)
        return features @ self.w


class LinearRegressorGD(Regression):
    def __init__(self) -> None:
        self.w = None

    def _add_constant(self, features: np.ndarray) -> np.ndarray:
        return np.hstack((features, np.ones((len(features), 1))))

    def learn(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            learning_rate: float = 1e-3,
            n_epochs: int = 500,
            random_state: int = 42,
    ) -> None:
        self._check_learn_shapes(features, targets)
        features = self._add_constant(features)

        rng = np.random.default_rng(random_state)
        self.w = rng.standard_normal(size=(features.shape[1],))

        for epoch in range(n_epochs):
            self.w = self.w - learning_rate * self._gradient(features, targets)
            if np.isnan(self.w).sum() > 0:
                raise ValueError("Weights have diverged", self.w)

    def _gradient(self, features: np.ndarray, targets: np.ndarray):
        return features.T @ features @ self.w - features.T @ targets

    def infer(self, features: np.ndarray) -> np.ndarray:
        self._check_infer_shapes(features)
        features = self._add_constant(features)
        return features @ self.w
