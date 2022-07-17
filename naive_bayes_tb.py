from sklearn import metrics as mt
from alekseiml.classification import NaiveBayesClassifier
from sklearn.datasets import load_files
from sklearn import feature_extraction
from sklearn.naive_bayes import MultinomialNB


def test_nb_by_filepath(train_path, test_path):
    nb = NaiveBayesClassifier()
    nb.learn(train_path)
    y, y_pred = nb.test(test_path)
    print('Accuracy: %2.2f %%' % (100. * mt.accuracy_score(y, y_pred)))


def test_sklearn_nb_by_filepath(train_path, test_path):
    twenty_train = load_files(train_path, encoding='latin1')
    twenty_test = load_files(test_path, encoding='latin1')

    vectorizer = feature_extraction.text.CountVectorizer()  # (max_features=len(clf.vocabulary))
    train_X, train_y = vectorizer.fit_transform(twenty_train.data), twenty_train.target
    test_X, test_y = vectorizer.transform(twenty_test.data), twenty_test.target
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    print("Training sklearn NP")
    sk_clf = MultinomialNB()
    sk_clf.fit(train_X, train_y)
    print("Training my own NB")
    clf = NaiveBayesClassifier()
    clf.learn_by_array(train_X, train_y)
    print("Testing sklearn NB")
    sk_pred_y = sk_clf.predict(test_X)
    print("Testing self-implemented NB")
    _, pred_y = clf.test_by_array(test_X, test_y)
    print('Accuracy for sklearn NB:             %2.2f %%' % (100. * mt.accuracy_score(test_y, sk_pred_y)))
    print('Accuracy for self-implemented NB:    %2.2f %%' % (100. * mt.accuracy_score(test_y, pred_y)))


# test_nb_by_filepath('data/newsgroups/20news-bydate-train', 'data/newsgroups/20news-bydate-test')
test_sklearn_nb_by_filepath('data/newsgroups/20news-bydate-train', 'data/newsgroups/20news-bydate-test')
