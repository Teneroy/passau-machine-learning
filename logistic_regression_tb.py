import sklearn
from sklearn import datasets as dt, feature_extraction, model_selection
from sklearn import linear_model as ln
from alekseiml.classification import LogisticRegression
import pandas as pd
import numpy as np


def sklearn_dataset_test():
    X, y = dt.make_classification(n_samples=1000, random_state=13)

    model = LogisticRegression()
    model.learn(X, y, n_epochs=100, learning_rate=1e-3, random_state=0)

    skl_model = ln.LogisticRegression(random_state=0)
    skl_model.fit(X, y)

    print(sklearn.metrics.roc_auc_score(y, model.infer(X)))
    print(sklearn.metrics.roc_auc_score(y, skl_model.predict_proba(X)[:, 1]))


def employee_dataset_test(path):
    data = pd.read_csv(path)
    print(data.columns)
    data = data.dropna()
    data2 = pd.get_dummies(data, columns=['Designation', 'Department'])
    X = np.ravel(data[['Department']].values)
    y = np.ravel(data[['Designation']].values)
    vectorizer = feature_extraction.text.CountVectorizer()
    X = vectorizer.

    # vectorizer2 = feature_extraction.text.CountVectorizer()
    # y = vectorizer.fit_transform(y).toarray()
    # t_fn = vectorizer.get_feature_names()
    # dt = data.columns.values.tolist()
    # X = vectorizer2.fit_transform(X)
    # X, y = np.array(vectorizer.fit_transform(X), dtype=np.float64), np.array(y, dtype=np.float64)

    model = LogisticRegression()
    model.learn(X, y, n_epochs=100, learning_rate=1e-3, random_state=0)

    skl_model = ln.LogisticRegression(random_state=0)
    skl_model.fit(X, y)

    print(sklearn.metrics.roc_auc_score(y, model.infer(X)))
    print(sklearn.metrics.roc_auc_score(y, skl_model.predict_proba(X)[:, 1]))


def diabetes_dataset_test(path):
    data = pd.read_csv(path)
    X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = data['Outcome']
    X /= np.linalg.norm(X)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

    model = LogisticRegression()
    model.learn(X_train, y_train, n_epochs=1000, learning_rate=1e-3, random_state=0)

    skl_model = ln.LogisticRegression(random_state=0)
    skl_model.fit(X_train, y_train)

    print(sklearn.metrics.roc_auc_score(y_test, model.infer(X_test)))
    print(sklearn.metrics.roc_auc_score(y_test, skl_model.predict_proba(X_test)[:, 1]))


# sklearn_dataset_test()
# employee_dataset_test('./data/Employee_monthly_salary.csv')
diabetes_dataset_test('./data/diabetes.csv')
