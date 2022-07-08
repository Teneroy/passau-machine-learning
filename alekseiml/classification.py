from base import Classifier
from abc import ABC
import re
import codecs
import os
import numpy as np
import math
from metrics import sigmoid
from random import shuffle


class RandomClassifier(Classifier):
    def __init__(self, num_classes: int = None):
        self._classes = np.arange(num_classes) if num_classes is not None and num_classes > 0 else None

    def learn(self, features, targets):
        observed_classes = np.unique(targets)
        if self._classes is None:
            self._classes = observed_classes
        for obs_clz in observed_classes:
            assert obs_clz in self._classes

    def infer(self, features):
        return np.random.choice(self._classes, size=features.shape[0])


class TrivialClassifier(Classifier):
    def __init__(self, num_classes: int = None):
        self._classes = np.arange(num_classes) if num_classes is not None and num_classes > 0 else None
        self._occurrences = {clz: 0 for clz in self._classes} if self._classes is not None else {}

    def learn(self, features, targets):
        observed_classes = np.unique(targets)
        if self._classes is None:
            self._classes = observed_classes
        for obs_clz in observed_classes:
            assert obs_clz in self._classes
        occurrences = np.bincount(targets.flatten())
        for clz, count in zip(self._classes, occurrences):
            if clz not in self._occurrences:
                self._occurrences[clz] = 0
            self._occurrences[clz] += count

    def infer(self, features):
        trivial_class = max(self._occurrences.items(), key=lambda x: x[1])[0]
        return np.array(np.ones(features.shape[0]) * trivial_class, dtype=np.int)


class NaiveBayesClassifier(Classifier):
    def __init__(self, min_count=1) -> None:
        self.min_count = min_count
        self.vocabulary = {}
        self.num_docs = 0
        self.classes = {}
        self.priors = {}
        self.conditionals = {}
        self.num_docs = 0

    def _build_vocab(self, path_newsgroups_train):
        for class_name in os.listdir(path_newsgroups_train):
            self.classes[class_name] = {"doc_counts": 0, "term_counts": 0, "terms": {}}
            print(class_name)
            path_class = os.path.join(path_newsgroups_train, class_name)
            for doc_name in os.listdir(path_class):
                terms = self._tokenize_file(os.path.join(path_class, doc_name))

                self.num_docs += 1
                self.classes[class_name]["doc_counts"] += 1

                for term in terms:
                    self.classes[class_name]["term_counts"] += 1
                    if term not in self.vocabulary:
                        self.vocabulary[term] = 1
                        self.classes[class_name]["terms"][term] = 1
                    else:
                        self.vocabulary[term] += 1
                        if term not in self.classes[class_name]["terms"]:
                            self.classes[class_name]["terms"][term] = 1
                        else:
                            self.classes[class_name]["terms"][term] += 1
        self.vocabulary = {k: v for k, v in self.vocabulary.items() if v > self.min_count}

    def _build_vocab_by_arrays(self, features, targets):
        i = 0
        for feature in features:
            class_name = targets[i]
            if self.classes.get(class_name) is None:
                self.classes[class_name] = {"doc_counts": 0, "term_counts": 0, "terms": {}}

            self.num_docs += 1
            self.classes[class_name]["doc_counts"] += 1

            terms = feature.indices
            occurrence = feature.data

            for j in range(len(terms)):
                term = terms[j]
                self.classes[class_name]["term_counts"] += 1
                self.vocabulary[term] = term
                if self.classes[class_name]["terms"].get(term) is None:
                    self.classes[class_name]["terms"][term] = occurrence[j]
                    continue
                self.classes[class_name]["terms"][term] = (self.classes[class_name]["terms"][term] + occurrence[j])

            i += 1

        self.vocabulary = {k: v for k, v in self.vocabulary.items() if v > self.min_count}

    def learn(self, path:str, targets=None):
        self._build_vocab(path)

        for cn in self.classes:
            self.priors[cn] = math.log(self.classes[cn]['doc_counts']) - math.log(self.num_docs)

            self.conditionals[cn] = {}
            cdict = self.classes[cn]['terms']
            c_len = sum(cdict.values())

            for term in self.vocabulary:
                t_ct = 1.
                t_ct += cdict[term] if term in cdict else 0.
                self.conditionals[cn][term] = math.log(t_ct) - math.log(c_len + len(self.vocabulary))

    def learn_by_array(self, features: np.array, targets: np.array):
        self._build_vocab_by_arrays(features, targets)
        for cn in self.classes:
            self.priors[cn] = math.log(self.classes[cn]['doc_counts']) - math.log(self.num_docs)

            self.conditionals[cn] = {}
            cdict = self.classes[cn]['terms']
            c_len = sum(cdict.values())

            for term in self.vocabulary:
                t_ct = 1.
                t_ct += cdict[term] if term in cdict else 0.
                self.conditionals[cn][term] = math.log(t_ct) - math.log(c_len + len(self.vocabulary))

    def test_by_array(self, features: np.array, targets: np.array):
        pred = []
        truth = []

        i = 0
        for feature in features:
            _, result_class = self._predict(feature.indices)
            pred.append(result_class)
            truth.append(targets[i])
            i += 1

        return truth, pred

    def test(self, path):
        pred = []
        truth = []

        for cls in self.classes:
            for file in os.listdir(os.path.join(path, cls)):
                doc_path = os.path.join(path, cls, file)
                _, result_class = self._predict(self._tokenize_file(doc_path))
                pred.append(result_class)
                truth.append(cls)

        return truth, pred

    def infer(self, document):
        token_list = self._tokenize_str(document)
        self._predict(token_list)

    def _predict(self, token_list):
        scores = {}
        for class_num, class_name in enumerate(self.classes):
            scores[class_name] = self.priors[class_name]
            for term in token_list:
                if term in self.vocabulary:
                    scores[class_name] += self.conditionals[class_name][term]
        return scores, max(scores, key=scores.get)

    def _tokenize_file(self, doc_file):
        with codecs.open(doc_file, encoding='latin1') as doc:
            doc = doc.read().lower()
            _header, _blankline, body = doc.partition('\n\n')
            return self._tokenize_str(body)  # return all words with #characters > 1

    def _tokenize_str(self, doc):
        return re.findall(r'\b\w\w+\b', doc)  # return all words with #characters > 1


class LogisticRegression(Classifier):
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
        # features = self._add_constant(features)

        rng = np.random.default_rng(random_state)
        self.w = rng.standard_normal(size=(features.shape[1] + 1,))

        for epoch in range(n_epochs):
            self.w = self.w - learning_rate * self._gradient(features, targets)
            if np.isnan(self.w).sum() > 0:
                raise ValueError("Weights have diverged", self.w)

    def _gradient(self, features: np.ndarray, targets: np.ndarray):
        return sum([(y_hat - y) * x for y_hat, y, x in zip(self.infer(features), targets, self._add_constant(features))])

    def infer(self, features: np.ndarray) -> np.ndarray:
        # self._check_infer_shapes(features)
        features = self._add_constant(features)
        y_pred = []
        for x in features:
            y_pred.append(sigmoid(self.w.T @ x))
        return np.array(y_pred)


class SVM():
    """Implementation of SVM with SGD"""

    def __init__(self, lmbd, D):
        self.lmbd = lmbd
        self.D = D + 1
        self.w = [0.] * self.D

    def sign(self, x):
        return -1. if x <= 0 else 1.

    def hinge_loss(self, target, y):
        return max(0, 1 - target * y)

    def data(self, test=False):
        if test:
            with open('test.csv', 'r') as f:
                samples = f.readlines()

                for t, row in enumerate(samples):

                    row = row.replace('\n', '')
                    row = row.split(',')

                    target = -1.

                    if row[3] == '1':
                        target = 1.
                    row = row[:3]

                    x = [float(c) for c in row] + [1.]  # inputs + bias

                    yield t, x, target

        else:

            with open('train.csv', 'r') as f:
                samples = f.readlines()
                shuffle(samples)

                for t, row in enumerate(samples):

                    row = row.replace('\n', '')
                    row = row.split(',')

                    target = -1.

                    if row[3] == '1':
                        target = 1.
                    row = row[:3]

                    x = [float(c) for c in row] + [1.]  # inputs + bias

                    yield t, x, target

    def train(self, x, y, alpha):
        if y * self.predict(x) < 1:

            for i in range(len(x)):
                self.w[i] = self.w[i] + alpha * ((y * x[i]) + (-2 * (self.lmbd) * self.w[i]))

        else:
            for i in range(len(x)):
                self.w[i] = self.w[i] + alpha * (-2 * (self.lmbd) * self.w[i])

        return self.w

    def predict(self, x):
        wTx = 0.
        for i in range(len(x)):
            wTx += self.w[i] * x[i]

        return wTx

    def learn(self, features, targets):
        tn = 0.
        tp = 0.
        total_positive = 0.
        total_negative = 0.
        accuracy = 0.
        loss = 0.
        last = 0
        t = 0
        for x, target in zip(features, targets):
            x = [float(c) for c in x] + [1.]
            if target == last:
                continue

            alpha = 1. / (self.lmbd * (t + 1.))
            w = self.train(x, target, alpha)
            last = target
            t += 1

        for x, target in zip(features, targets):
            x = [float(c) for c in x] + [1.]
            pred = self.predict(x)
            loss += self.hinge_loss(target, pred)

            pred = self.sign(pred)

            if target == 1:
                total_positive += 1.
            else:
                total_negative += 1.

            if pred == target:
                accuracy += 1.
                if pred == 1:
                    tp += 1.
                else:
                    tn += 1.

        loss = loss / (total_positive + total_negative)
        acc = accuracy / (total_positive + total_negative)

        return loss, acc, tp / total_positive, tn / total_negative, w



    def fit(self):
        test_count = 0.

        tn = 0.
        tp = 0.

        total_positive = 0.
        total_negative = 0.

        accuracy = 0.
        loss = 0.

        last = 0
        w = 0
        for t, x, target in self.data(test=False):

            if target == last:
                continue

            alpha = 1. / (self.lmbd * (t + 1.))
            w = self.train(x, target, alpha)
            last = target

        for t, x, target in self.data(test=True):

            pred = self.predict(x)
            loss += self.hinge_loss(target, pred)

            pred = self.sign(pred)

            if target == 1:
                total_positive += 1.
            else:
                total_negative += 1.

            if pred == target:
                accuracy += 1.
                if pred == 1:
                    tp += 1.
                else:
                    tn += 1.

        loss = loss / (total_positive + total_negative)
        acc = accuracy / (total_positive + total_negative)

        # print 'Loss', loss, '\nTrue Negatives', tn/total_negative * 100, '%', '\nTrue Positives', tp/total_positive * 100, '%','\nPrecision', accuracy/(total_positive+total_negative) * 100, '%', '\n'

        return loss, acc, tp / total_positive, tn / total_negative, w
