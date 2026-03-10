from sklearn.linear_model import LogisticRegression

def get_logistic_regression(max_iter=1000):
    """
    Returns an instantiated Logistic Regression classifier.
    """
    return LogisticRegression(max_iter=max_iter, solver='lbfgs', multi_class='multinomial')
