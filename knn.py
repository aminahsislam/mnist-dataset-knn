import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import sklearn

digits = load_digits()
neighbors_settings = range(1, 11)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for test, ax in zip([0.1, 0.2, 0.3], axes):
    training_accuracy = []
    test_accuracy = []
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target,test_size=test, random_state=66)
    for n_neighbors in neighbors_settings:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        training_accuracy.append(clf.score(X_train, y_train))
        test_accuracy.append(clf.score(X_test, y_test))
    ax.plot(neighbors_settings, training_accuracy, color='blue', label="Training Accuracy")
    ax.plot(neighbors_settings, test_accuracy, color='red', label="Test Accuracy")
    ax.set_xlabel("Number of Neighbors")
axes[0].legend(["Training accuracy", "Test accuracy"], loc="best")

plt.show()



