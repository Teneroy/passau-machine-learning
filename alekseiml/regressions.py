from base import  Regression, BaseModel
import numpy as np


class RandomNormalEstimator(BaseModel):
    def __init__(self):
        self._mean = 0
        self._std = 1

    def learn(self, features, targets):
        self._check_learn_shapes(features, targets)
        self._mean = np.mean(targets)
        self._std = np.std(targets)

    def infer(self, features):
        return np.random.normal(self._mean, self._std, (features.shape[0],))


class RandomUniformEstimator(BaseModel):
    def __init__(self):
        self._tmin = 0
        self._tmax = 1

    def learn(self, features, targets):
        self._check_learn_shapes(features, targets)
        self._tmin = min(self._tmin, np.min(targets))
        self._tmax = max(self._tmax, np.max(targets))

    def infer(self, features):
        return np.random.uniform(self._tmin, self._tmax, (features.shape[0],))


class LinearRegressionLMS(BaseModel):
    def __init__(self) -> None:
        self.w = None

    def _add_constant(self, features: np.ndarray) -> np.ndarray:
        return np.hstack((features, np.ones((len(features), 1))))

    def learn(self, features: np.ndarray, targets: np.ndarray) -> None:
        # self._check_learn_shapes(features, targets)
        features = self._add_constant(features)
        self.w = np.linalg.inv(features.T @ features) @ features.T @ targets  # there is also np.linalg.pinv

    def infer(self, features: np.ndarray) -> np.ndarray:
        # self._check_infer_shapes(features)
        features = self._add_constant(features)
        return features @ self.w


class LinearRegressorGD(BaseModel):
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
        # self._check_learn_shapes(features, targets)
        features = self._add_constant(features)

        rng = np.random.default_rng(random_state)
        self.w = rng.standard_normal(size=(features.shape[1],))

        for epoch in range(n_epochs):
            # print(epoch)
            gd = self._gradient(features, targets)
            self.w = self.w - learning_rate * self._gradient(features, targets)

    def _gradient(self, features: np.ndarray, targets: np.ndarray):
        return features.T @ features @ self.w - (features.T @ targets)

    def infer(self, features: np.ndarray) -> np.ndarray:
        # self._check_infer_shapes(features)
        features = self._add_constant(features)
        return features @ self.w


class LinearRegressorGD2(Regression):
    def __init__(self) -> None:
        self.w = None

    def _add_constant(self, X: np.ndarray) -> np.ndarray:
        """Add a constant column to a matrix.

        Args:
            X (np.ndarray): Original data matrix

        Returns:
            np.ndarray: Original data matrix with concatenated column of all ones.
        """
        return np.hstack((X, np.ones((len(X), 1))))

    def learn(
            self,
            X: np.ndarray,
            y: np.ndarray,
            learning_rate: float = 1e-3,
            n_epochs: int = 500,
            random_state: int = 42,
    ) -> None:
        """Fit the parameters of the model to the data with gradient descent.

        Args:
            X (np.ndarray): features
            y (np.ndarray): targets
            learning_rate (float): step size of gradient descent
            n_epochs (int): number of parameter updates
            random_state (int): seed for reproducibility
        """
        # modify the features such that a bias can be learned easily
        X = self._add_constant(X)

        # initialize randomly
        rng = np.random.default_rng(random_state)
        self.w = rng.standard_normal(size=(X.shape[1],))

        # gradient descent
        for _ in range(n_epochs):
            self.w = self.w - learning_rate * self._gradient(X, y)

    def _gradient(self, X: np.ndarray, y: np.ndarray):
        return X.T @ X @ self.w - X.T @ y

    def infer(self, X: np.ndarray) -> np.ndarray:
        """Use parameters to predict values

        Args:
            X (np.ndarray): features

        Returns:
            np.ndarray: predicted targets
        """
        X = self._add_constant(X)
        return X @ self.w