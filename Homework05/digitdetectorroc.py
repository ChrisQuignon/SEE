#!/usr/bin/env python
import numpy as np
import pandas as pd
#from copper.ml.nn.nn import NN
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

# fom http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

    # plt.figure(figsize = (8, 6))
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)


    fig = plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap=cmap)

    plt.title(title + '\n Confusion matrix')
    plt.colorbar(im, use_gridspec=True, orientation="vertical", shrink=0.75)
    tick_marks =labels
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('img/' + title +'.png')

#Load procedure from https://github.com/danielfrg/kaggle-digits/blob/master/v1/nn-whole-mnist.ipynb
f = gzip.open('data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#split sets
x_train, y_train = train_set[0], train_set[1]
x_valid, y_valid = valid_set[0], valid_set[1]
x_test, y_test = test_set[0], test_set[1]

#Binarize labels
lb = LabelBinarizer()
lb.fit(range(10))

y_train = lb.transform(y_train)
n_train_classes = y_train.shape[1]

y_test = lb.transform(y_test)
n_test_classes = y_test.shape[1]

y_valid = lb.transform(y_valid)
n_valid_classes = y_valid.shape[1]

#PREPROCESSING
def pxl_histogram(x):
    img = x.reshape(x.shape[0], 28, 28)
    img = img[:, 3:-3, 3:-3]#cut off frame
    horizontal_hist = np.sum(img, axis=1)
    vertical_hist = np.sum(img, axis=2)
    img = np.hstack([horizontal_hist, vertical_hist])
    return img

x_train = pxl_histogram(x_train)
x_valid = pxl_histogram(x_valid)
x_test = pxl_histogram(x_test)


#targets
fpr = dict()
tpr = dict()
roc_auc = dict()

#classifications
clfs = []
clfs.append(DecisionTreeClassifier(random_state=0))
clfs.append(RandomForestClassifier())
clfs.append(BernoulliNB())
clfs.append(MultinomialNB())
clfs.append(AdaBoostClassifier())
# clfs.append(GradientBoostingClassifier()) #runs too long
# clfs.append(KNeighborsClassifier(10)) #runs too long
# clfs.append(svm.SVC()) #runs too long

for clf in clfs:
    n =  clf.__class__.__name__
    print ''
    print clf, ':'

    #binray classification is needed for ROCs
    classifier = OneVsRestClassifier(clf)
    classifier.fit(x_train, y_train)

    pbs = classifier.predict_proba(x_test)
    prediction = classifier.predict(x_test)
    prediction = lb.inverse_transform(prediction)


    #SAVE
    classifier = np.save('saves/' + n + 'classifier.np', classifier)
    prediction.tofile('saves/' + n + 'prediction.np')
    pbs.tofile('saves/' + n + 'probabilities.np')

    # #LOAD
    # classifier = np.load('saves/' + n + 'classifier.np.npy')
    # classifier = classifier.item()
    # prediction = np.fromfile('saves/' + n + 'prediction.np')
    # pbs = np.fromfile('saves/' + n + 'probabilities.np')

    #PRINT REPORTS
    print classification_report(lb.inverse_transform(y_test), prediction)
    print 'Confusion matrix:'
    cm =  confusion_matrix(lb.inverse_transform(y_test), prediction)
    print cm
    plot_confusion_matrix(cm, title = n )
    print ''
    print ''

    # Compute ROC curve
    fpr[n], tpr[n], _ = roc_curve(y_test.ravel(), pbs.ravel())
    roc_auc[n] = auc(fpr[n], tpr[n])

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr[n], tpr[n],
             label= n + ' ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc[n]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(n + ' ROC')
    plt.legend(loc="lower right")
    plt.savefig('img/' + n + 'roc.png')
    # plt.show()

#PLOT ALL ROCS ON ONE DIAGRAM
plt.figure()
for k in fpr.keys():

    # Plot ROC curve
    plt.plot(fpr[k], tpr[k],
             label= k )#+ ' ROC curve (area = {0:0.2f})'
                   #''.format(roc_auc[k]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROCs')
plt.legend(loc="lower right")
plt.savefig('img/all_roc.png')
# plt.show()
