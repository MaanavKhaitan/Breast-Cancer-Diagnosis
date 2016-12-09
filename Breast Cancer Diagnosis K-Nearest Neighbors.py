#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 11:10:34 2016

@author: maaanavkhaitan
"""

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.decomposition import PCA
from sklearn import manifold

Test_PCA = True


def plotDecisionBoundary(model, X, y):
  print "Plotting..."
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()



X = pd.read_csv('/Users/maaanavkhaitan/Downloads/DAT210x-master/Module5/Datasets/breast-cancer-wisconsin.data')
X.columns = ['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']

Y = X['status']
X = X.drop(labels=['status', 'sample'], axis=1)

X.nuclei = pd.to_numeric(X.nuclei, errors='coerce')
X = X.fillna(X.mean())
X.nuclei = X.nuclei.fillna(3.5)

T = preprocessing.StandardScaler().fit_transform(X)
#T = preprocessing.MinMaxScaler().fit_transform(X)
#T = preprocessing.MaxAbsScaler().fit_transform(X)
#T = preprocessing.Normalizer().fit_transform(X)
T = X # No Change


data_train, data_test, label_train, label_test = train_test_split(X, Y, test_size=0.33, random_state=7)


model = None
if Test_PCA:
  print "Computing 2D Principle Components"
  model = PCA(n_components=2)
else:
  print "Computing 2D Isomap Manifold"  
  model = manifold.Isomap(n_neighbors=5, n_components=2)



model.fit(data_train)
data_train = model.transform(data_train)
data_test = model.transform(data_test)


model = KNeighborsClassifier(n_neighbors=11, weights='distance', p=2)
model.fit(data_train, label_train)
print model.score(data_test, label_test)

plotDecisionBoundary(model, data_test, label_test)

print model.predict((-15,6))
