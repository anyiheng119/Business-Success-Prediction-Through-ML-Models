# -*- coding: utf-8 -*-
"""
Evaluations
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
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

# Splitting 3 types of features
ex_rate_features = numerical_feature.iloc[:,4:7] # 3 exchange rate features
web_features = numerical_feature.iloc[:,-4:]     # 4 web search features
company_feature = numerical_feature.iloc[:,0:4]

#####################################################################################
# Normalization for numerical features
# only company-based features #######################################################
scaler = sklearn.preprocessing.StandardScaler() # z-score
X_normal = DataFrame(scaler.fit_transform(company_feature))
X_normal = DataFrame(pd.concat([onehot_feature,X_normal],axis=1)) # conbine one-hot features with normalized numerical features


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

#########################
# KNN model 
#########################
model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2)
model_knn.fit(X_train_res, Y_train_res.values.ravel())

model_knn.score(X_test, Y_test)                                        # 0.802784
f1_score(Y_test, model_knn.predict(X_test),average="weighted")         # 0.815424
func_calConfusionMatrix(model_knn.predict(X_test),Y_test)

#########################
# SVM model 
#########################
model_svm = svm.SVC(kernel='linear', C=38, max_iter = 100)
model_svm.fit(X_train_res, Y_train_res.values.ravel())
model_svm.score(X_test, Y_test) # 0.10608404
f1_score(Y_test, model_svm.predict(X_test),average="weighted") #0.0253808

#########################
# Random Forest 
#########################
clf_rf = RandomForestClassifier(n_estimators=111,max_depth=29, random_state=1)
clf_rf.fit(X_train_res, Y_train_res.values.ravel())
clf_rf.score(X_test, Y_test)              # 0.7370456
f1_score(Y_test, clf_rf.predict(X_test),average="weighted")  # 0.7788285
func_calConfusionMatrix(clf_rf.predict(X_test),Y_test) 

#########################
# Logistic model
#########################

LogReg = LogisticRegression(C=8101)
LogReg.fit(X_train_res, Y_train_res.values.ravel())
LogReg.score(X_test, Y_test)                                     # 0.509925
f1_score(Y_test, LogReg.predict(X_test),average="weighted")      # 0.595488
func_calConfusionMatrix(LogReg.predict(X_test),Y_test)


#####################################################################################
# Normalization for numerical features
# company-based features + exchange rate ############################################
scaler = sklearn.preprocessing.StandardScaler() # z-score
com_and_ex = pd.concat([company_feature,ex_rate_features],axis=1)
X_normal = DataFrame(scaler.fit_transform(com_and_ex))
X_normal = DataFrame(pd.concat([onehot_feature,X_normal],axis=1)) # conbine one-hot features with normalized numerical features


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


X_train_res.info()
#########################
# KNN model 
#########################
model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2)
model_knn.fit(X_train_res, Y_train_res.values.ravel())

model_knn.score(X_test, Y_test)                                        # 0.811291
f1_score(Y_test, model_knn.predict(X_test),average="weighted")         # 0.824290
func_calConfusionMatrix(model_knn.predict(X_test),Y_test)

#########################
# SVM model 
#########################
model_svm = svm.SVC(kernel='linear', C=38, max_iter = 100)
model_svm.fit(X_train_res, Y_train_res.values.ravel())
model_svm.score(X_test, Y_test)                                # 0.89468935
f1_score(Y_test, model_svm.predict(X_test),average="weighted") # 0.84964734

#########################
# Random Forest 
#########################
clf_rf = RandomForestClassifier(n_estimators=111,max_depth=29, random_state=1)
clf_rf.fit(X_train_res, Y_train_res.values.ravel())
clf_rf.score(X_test, Y_test)                                 # 0.84209847
f1_score(Y_test, clf_rf.predict(X_test),average="weighted")  # 0.8464005
func_calConfusionMatrix(clf_rf.predict(X_test),Y_test) 

#########################
# Logistic model
#########################

LogReg = LogisticRegression(C=8101)
LogReg.fit(X_train_res, Y_train_res.values.ravel())
LogReg.score(X_test, Y_test)                                     # 0.6122712
f1_score(Y_test, LogReg.predict(X_test),average="weighted")      # 0.6872931
func_calConfusionMatrix(LogReg.predict(X_test),Y_test)
