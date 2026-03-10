import sys, os
sys.path.insert(0, os.path.abspath('.'))
from src.data_loader import get_filtered_mnist, get_flat_numpy_arrays
from src.svd_classifier import SVDRankKClassifier, SVDFullRankClassifier
from src.baselines import get_logistic_regression
from src.cnn_model import CNNClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

train_dataset, test_dataset = get_filtered_mnist(digits=(3, 8))
X_train, y_train, X_test, y_test = get_flat_numpy_arrays(train_dataset, test_dataset)

# SVD rank-10
m1 = SVDRankKClassifier(k=10)
m1.fit(X_train, y_train)
p1 = m1.predict(X_test)
print("SVD rank-10 Acc:", accuracy_score(y_test, p1), "3->8:", confusion_matrix(y_test, p1, labels=[3,8])[0,1])

# SVD full-rank
m2 = SVDFullRankClassifier()
m2.fit(X_train, y_train)
p2 = m2.predict(X_test)
print("SVD full-rank Acc:", accuracy_score(y_test, p2), "3->8:", confusion_matrix(y_test, p2, labels=[3,8])[0,1])

# CNN
cnn = CNNClassifier(epochs=1, batch_size=64)
cnn.fit(train_dataset)
p3 = cnn.predict(test_dataset)
print("CNN Acc:", accuracy_score(y_test, p3), "3->8:", confusion_matrix(y_test, p3, labels=[3,8])[0,1])

