

import numpy as np
import pandas as pd
#from copper.ml.nn.nn import NN
import cPickle, gzip, numpy
import pylab
from sklearn import svm

#Load procedure from https://github.com/danielfrg/kaggle-digits/blob/master/v1/nn-whole-mnist.ipynb
f = gzip.open('data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

x_train, y_train = train_set[0], train_set[1]
x_valid, y_valid = valid_set[0], valid_set[1]
x_test, y_test = test_set[0], test_set[1]

# X = np.vstack((X_train, X_valid, X_test))
# y = np.hstack((y_train, y_valid, y_test))



def pxl_histogram(x):
    #create histograms

    img = x.reshape(x.shape[0], 28, 28)
    img = img[:, 3:-3, 3:-3]#cut off frame
    horizontal_hist = np.sum(img, axis=1)
    vertical_hist = np.sum(img, axis=2)
    img = np.hstack([horizontal_hist, vertical_hist])
    return img


clf = svm.SVC()
clf.fit(pxl_histogram(x_train), y_train)
print clf.predict(pxl_histogram(X_test))
