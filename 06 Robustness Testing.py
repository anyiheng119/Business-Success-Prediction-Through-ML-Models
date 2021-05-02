# -*- coding: utf-8 -*-
"""
Robustness testing

"""

import numpy as np
import pandas as pd
import seaborn as sb
from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
import sklearn

from sklearn import preprocessing
import sklearn.svm as svm
from confusion_matrix import func_calConfusionMatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score, f1_score, precision_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore') 

# import data
data = DataFrame(read_csv("clean_data_hot.csv"))
data = data.drop("Unnamed: 0", axis=1)
data.info()

target = DataFrame(data["label"])
X = data.drop(columns="label")
onehot_feature = X.iloc[:,0:59]
numerical_feature = X.iloc[:,59:]

# Normalization for numerical features
scaler = sklearn.preprocessing.StandardScaler() # z-score
X_normal = DataFrame(scaler.fit_transform(numerical_feature))
X_normal = DataFrame(pd.concat([onehot_feature,X_normal],axis=1)) # conbine one-hot features with normalized numerical features
X_normal.columns = X.columns

# data splitting 
X_else, X_test = train_test_split(X_normal, test_size=0.4, random_state=1)            # 40% testing samples
X_train, X_validation= train_test_split(X_else, test_size=1/6, random_state=1) # 10% validation samples

Y_else, Y_test = train_test_split(target, test_size=0.4, random_state=1)       # 50% training samples
Y_train, Y_validation= train_test_split(Y_else, test_size=1/6, random_state=1)

# Oversampling on only the training set
sm = SMOTE(random_state=12, sampling_strategy = 1.0)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)

Y_train['label'].value_counts()
Y_train_res['label'].value_counts()

# half set (50%)
removed, X_50 = train_test_split(X_normal, test_size=0.5, random_state=1) 
removed, Y_50 = train_test_split(target, test_size=0.5, random_state=1) 

# 1/10 set (10%)
removed, X_10 = train_test_split(X_normal, test_size=0.1, random_state=1) 
removed, Y_10 = train_test_split(target, test_size=0.1, random_state=1) 

####################
# KNN
####################
model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2)
model_knn.fit(X_train_res, Y_train_res.values.ravel())

# 5fold CV: 100% set
np.mean(cross_val_score(model_knn, X_normal, target, cv=5))                        # 0.883217
np.mean(cross_val_score(model_knn, X_normal, target, scoring="f1_weighted",cv=5))  # 0.848880

# 5fold CV: 50% set
np.mean(cross_val_score(model_knn, X_50, Y_50, cv=5))                       # 0.8885330
np.mean(cross_val_score(model_knn, X_50, Y_50, scoring="f1_weighted",cv=5)) # 0.8559930

# 5fold CV: 10% set
np.mean(cross_val_score(model_knn, X_10, Y_10, cv=5))                       # 0.8958762
np.mean(cross_val_score(model_knn, X_10, Y_10, scoring="f1_weighted",cv=5)) # 0.8654869


####################
# SVM model
####################
model_svm = svm.SVC(kernel='linear', C=38, max_iter = 100)
model_svm.fit(X_train_res, Y_train_res.values.ravel())

# 5fold CV: 100% set
np.mean(cross_val_score(model_svm, X_normal, target, cv=5))                        # 0.631399
np.mean(cross_val_score(model_svm, X_normal, target, scoring="f1_weighted",cv=5))  # 0.613479

# 5fold CV: 50% set
np.mean(cross_val_score(model_svm, X_50, Y_50, cv=5))                       # 0.171685
np.mean(cross_val_score(model_svm, X_50, Y_50, scoring="f1_weighted",cv=5)) # 0.158009

# 5fold CV: 10% set
np.mean(cross_val_score(model_svm, X_10, Y_10, cv=5))                       # 0.5134020
np.mean(cross_val_score(model_svm, X_10, Y_10, scoring="f1_weighted",cv=5)) # 0.5526565


#########################
# Logistic model
#########################
LogReg = LogisticRegression(C=8101)
LogReg.fit(X_train_res, Y_train_res.values.ravel())

# 5fold CV: 100% set
np.mean(cross_val_score(LogReg, X_normal, target, cv=5))                        # 0.891312
np.mean(cross_val_score(LogReg, X_normal, target, scoring="f1_weighted",cv=5))  # 0.842289

# 5fold CV: 50% set
np.mean(cross_val_score(LogReg, X_50, Y_50, cv=5))                       # 0.895133
np.mean(cross_val_score(LogReg, X_50, Y_50, scoring="f1_weighted",cv=5)) # 0.848617

# 5fold CV: 10% set
np.mean(cross_val_score(LogReg, X_10, Y_10, cv=5))                       # 0.893298
np.mean(cross_val_score(LogReg, X_10, Y_10, scoring="f1_weighted",cv=5)) # 0.849172

#########################
# Random Forest 
#########################
clf_rf = RandomForestClassifier(n_estimators=111,max_depth=29, random_state=1)
clf_rf.fit(X_train_res, Y_train_res.values.ravel())

# 5fold CV: 100% set
np.mean(cross_val_score(clf_rf, X_normal, target, cv=5))                        # 0.876927
np.mean(cross_val_score(clf_rf, X_normal, target, scoring="f1_weighted",cv=5))  # 0.853586

# 5fold CV: 50% set
np.mean(cross_val_score(clf_rf, X_50, Y_50, cv=5))                       # 0.881727
np.mean(cross_val_score(clf_rf, X_50, Y_50, scoring="f1_weighted",cv=5)) # 0.859173

# 5fold CV: 10% set
np.mean(cross_val_score(clf_rf, X_10, Y_10, cv=5))                       # 0.8819587
np.mean(cross_val_score(clf_rf, X_10, Y_10, scoring="f1_weighted",cv=5)) # 0.8602740



