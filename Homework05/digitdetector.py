#!/usr/bin/env python
import numpy as np
import pandas as pd
#from copper.ml.nn.nn import NN
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

#Load procedure from https://github.com/danielfrg/kaggle-digits/blob/master/v1/nn-whole-mnist.ipynb
f = gzip.open('data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

x_train, y_train = train_set[0], train_set[1]
x_valid, y_valid = valid_set[0], valid_set[1]
x_test, y_test = test_set[0], test_set[1]

def pxl_histogram(x):
    img = x.reshape(x.shape[0], 28, 28)
    img = img[:, 3:-3, 3:-3]#cut off frame
    horizontal_hist = np.sum(img, axis=1)
    vertical_hist = np.sum(img, axis=2)
    img = np.hstack([horizontal_hist, vertical_hist])
    return img

clf = svm.SVC(verbose=True)
clf.fit(pxl_histogram(x_train), y_train)
prediction =  clf.predict(pxl_histogram(x_test))
print ''
cm = confusion_matrix(y_test, prediction)
# print cm


# fom http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

    # plt.figure(figsize = (8, 6))
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)


    fig = plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap=cmap)

    plt.title(title)
    plt.colorbar(im, use_gridspec=True, orientation="vertical", shrink=0.75)
    tick_marks =labels
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('img/' + title +'.png')

# Compute confusion matrix
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
