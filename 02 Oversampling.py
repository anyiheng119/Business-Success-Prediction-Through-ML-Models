# -*- coding: utf-8 -*-
"""
Oversample
by Yiheng An

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

# plot the correlation between the features
plt.figure(figsize=(15, 10))
corr = numerical_feature.corr()
corr = sb.heatmap(corr, center=0, annot=True, cmap='YlGnBu')
plt.show()

# Normalization for numerical features
#scaler = sklearn.preprocessing.StandardScaler()
#X = DataFrame(scaler.fit_transform(X))
#X = pd.concat([onehot_feature,X],axis=1) # conbine one-hot features with normalized numerical features

# data splitting 
X_else, X_test = train_test_split(X, test_size=0.4, random_state=1)            # 40% testing samples
X_train, X_validation= train_test_split(X_else, test_size=1/6, random_state=1) # 10% validation samples

Y_else, Y_test = train_test_split(target, test_size=0.4, random_state=1)       # 50% training samples
Y_train, Y_validation= train_test_split(Y_else, test_size=1/6, random_state=1)


# Oversampling on only the training set
sm = SMOTE(random_state=12, sampling_strategy = 1.0)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)

Y_train['label'].value_counts()
Y_train_res['label'].value_counts()

# plot the oversampled data
import mglearn
X_train.info()

mglearn.discrete_scatter(X_train["IPO_growth_5years"], X_train["MA_growth_5years"],Y_train["label"],alpha = 0.6,s=8)
plt.xlabel("IPO_growth_5years")
plt.ylabel("MA_growth_5years")
plt.legend()
plt.show()

mglearn.discrete_scatter(X_train_res["IPO_growth_5years"], X_train_res["MA_growth_5years"],Y_train_res["label"],alpha = 0.6,s=8)
plt.xlabel("IPO_growth_5years")
plt.ylabel("MA_growth_5years")
plt.legend()
plt.show()

