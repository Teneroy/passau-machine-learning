from abc import ABC
import re
import codecs
import os
from base import Classifier, BaseModel
import numpy as np
import math


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
    def __int__(self, min_count=1):
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

    def learn(self, path, targets=None):
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