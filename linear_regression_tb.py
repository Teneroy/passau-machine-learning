import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

from alekseiml.regressions import LinearRegressionLMS, LinearRegressorGD, RandomNormalEstimator, RandomUniformEstimator
import pandas as pd
from sklearn import metrics as skmt
from sklearn import datasets as dts


def test_random_normal_estimator(X, y):
    random_normal = RandomNormalEstimator()
    random_normal.learn(X, y)
    pred_norm = random_normal.infer(X)
    return pred_norm


def test_random_uniform_estimator(X, y):
    random_uniform = RandomUniformEstimator()
    random_uniform.learn(X, y)
    pred_uniform = random_uniform.infer(X)
    return pred_uniform


def test_linear_regression_lms(X, y):
    lms = LinearRegressionLMS()
    lms.learn(X, y)
    pred_lms = lms.infer(X)
    return pred_lms


def test_linear_regression_gd(X, y):
    gd = LinearRegressorGD()
    gd.learn(X, y)
    pred_gd = gd.infer(X)
    return pred_gd


def test_regressions(X, y):
    return {
        'linear': test_linear_regression_lms(X, y),
        'gd': test_linear_regression_gd(X, y),
        'uniform': test_random_uniform_estimator(X, y),
        'normal': test_random_normal_estimator(X, y)
    }


def plot_regressions(X, y):
    reg_results = test_regressions(X, y)
    plt.plot(X, y, 'o')
    print('linear regression red')
    print('linear regression with gd green')
    print('uniform estimator blue')
    print('normal estimator orange')
    plt.plot(X, reg_results['linear'], '-r')
    plt.plot(X, reg_results['gd'], '-g')
    plt.plot(X, reg_results['uniform'], '-b')
    plt.plot(X, reg_results['normal'], '-p')
    plt.show()


def predict_age_by_credit_car_dept(path):
    data = pd.read_csv(path)
    print(data.columns)
    X = np.array(data[['Tenure_in_org_in_months', 'Age']].values, dtype=np.float64)
    y = np.array(np.ravel(data[['GROSS']].values), dtype=np.float64)
    X /= np.linalg.norm(X)
    y /= np.linalg.norm(y)
    plt.plot(X, y, 'o')
    # gd = LinearRegressionLMS()
    gd = LinearRegressorGD()
    sk_gd = sklearn.linear_model.LinearRegression()
    # X, y, true_coefs = dts.make_regression(n_samples=100, n_features=2, n_informative=3, random_state=0,
    #                                                     coef=True, noise=10)
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0, train_size=0.7)

    gd.learn(X, y, n_epochs=1000000)
    # gd.learn(X, y)
    sk_gd.fit(X, y)
    pred_gd = gd.infer(X)

    plt.plot(X, pred_gd, '-r')
    plt.show()
    print(skmt.r2_score(y, pred_gd))
    print(skmt.r2_score(y, sk_gd.predict(X)))


data = np.array([[5,30530, 50],[7,90000, 79],[15,159899, 124],[28,270564, 300]])
X = data[:,[0]]
y = data[:,[1]]
print ("Independent variables:", X, type(X))
print ("Dependent variable:", y, type(y))

predict_age_by_credit_car_dept('./data/Employee_monthly_salary.csv')
