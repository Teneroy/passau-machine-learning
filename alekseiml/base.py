from abc import ABCMeta
import numpy as np
import abc


class BaseModel:
    @abc.abstractmethod
    def learn(self, features: np.array, targets: np.array):
        """
        :param features: numpy array of shape (N, d) with N being the number of samples and d being the number of feature dimensions
        :param targets: numpy array of shape (N, 1) with N being the number of samples as in the provided features and 1 being the number of target dimensions
        :return:
        """
        pass

    @abc.abstractmethod
    def infer(self, features):
        """
        :param features: np array of shape (N, d)
        :return:
        """
        pass

    def _pearson(self, y0, y1):
        return np.cov(y0.flatten(), y1.flatten()) / (np.std(y0.flatten(), ddof=1) * np.std(y1.flatten(), ddof=1))

    @abc.abstractmethod
    def _check_feature_shapes(self, features: np.array):
        pass

    @abc.abstractmethod
    def _check_learn_shapes(self, features: np.array, targets: np.array):
        pass

    @abc.abstractmethod
    def _check_infer_shapes(self, features: np.array):
        pass


class Classifier(BaseModel, metaclass=ABCMeta):
    def _check_feature_shapes(self, features: np.array):
        pass

    def _check_learn_shapes(self, features: np.array, targets: np.array):
        pass

    def _check_infer_shapes(self, features: np.array):
        pass


class Regression(BaseModel, metaclass=ABCMeta):
    def _check_feature_shapes(self, features: np.array):
        assert features is not None
        assert len(features.shape) == 2
        assert features.shape[0] > 0
        assert features.shape[1] > 0

    def _check_learn_shapes(self, features: np.array, targets: np.array):
        self._check_feature_shapes(features)
        assert targets is not None
        assert len(targets.shape) == 1
        assert targets.shape[0] == features.shape[0]

    def _check_infer_shapes(self, features: np.array):
        self._check_feature_shapes(features)


class Clusterisation(BaseModel, metaclass=ABCMeta):
    def _check_feature_shapes(self, features: np.array):
        pass

    def _check_learn_shapes(self, features: np.array, targets: np.array):
        pass

    def _check_infer_shapes(self, features: np.array):
        pass
