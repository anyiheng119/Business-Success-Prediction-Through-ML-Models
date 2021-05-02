# -*- coding: utf-8 -*-
"""
Prameter tuning: KNN, SVM, Logistic and Random Forest 

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

#########################
# KNN model (need nromalization)
#########################

# Parameter tunning over validation set
neighbors = []
accracy = []
precision = []
f1 = []
for i in range(1, 50):
    model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i)
    model_knn.fit(X_train_res, Y_train_res.values.ravel())
    accracy_score = model_knn.score(X_validation, Y_validation)
    Precision = precision_score(Y_validation, model_knn.predict(X_validation))
    F1 = f1_score(Y_validation, model_knn.predict(X_validation), average='weighted')
    neighbors.append(i)
    accracy.append(accracy_score)
    precision.append(Precision)
    f1.append(F1)

df = {'neighbors':neighbors, 'accracy':accracy, 'precision':precision, 'f1': f1}
df = pd.DataFrame(df)
print(df[df['precision'] == max(precision)])  
# neighbors   accracy  precision      f1
#    6       0.771134   0.263658  0.795108

print(df[df['accracy'] == max(accracy)])
# neighbors   accracy   precision     f1
#    2       0.810309   0.244813  0.80964

# Plot the tuning process
plt.plot(neighbors, accracy)
plt.title('KNN model over validation set')
plt.xlabel('neighbors')
plt.ylabel('accracy')
plt.show()

plt.plot(neighbors, f1)
plt.title('KNN model over validation set')
plt.xlabel('neighbors')
plt.ylabel('f1 score')
plt.show()

# Test results
model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2)
model_knn.fit(X_train_res, Y_train_res.values.ravel())

model_knn.score(X_test, Y_test)                                        # 0.8259860
f1_score(Y_test, model_knn.predict(X_test),average="weighted")         # 0.83593871
func_calConfusionMatrix(model_knn.predict(X_test),Y_test)

#########################
# SVM model (need nromalization)
#########################

# Parameter tunning: kernel types
c_range =  range(1,100,1)
kernel_types = ['linear', 'poly', 'rbf']
accracy_svm = []
f1_svm = []
k_type = []
c = []

for c_value in c_range:
    for kernel_value in kernel_types:
        model = svm.SVC(kernel=kernel_value, C=c_value, max_iter = 100)  # aim to save time
        model.fit(X_train_res, Y_train_res.values.ravel())
        accracy_score = model.score(X_validation, Y_validation)
        F1 = f1_score(Y_validation, model.predict(X_validation),average="weighted")
        k_type.append(kernel_value)
        c.append(c_value)
        accracy_svm.append(accracy_score)
        f1_svm.append(F1)


df = {'kernel_types':k_type, 'c':c, 'accracy':accracy_svm, 'f1': f1_svm}
df = pd.DataFrame(df)
print(df[df['f1'] == max(f1_svm)])  
# kernel_types   c   accracy        f1
#    linear     38  0.869588  0.818282

# Test results
model_svm = svm.SVC(kernel='linear', C=38, max_iter = 100)
model_svm.fit(X_train_res, Y_train_res.values.ravel())
model_svm.score(X_test, Y_test)                                     # 0.89082237
f1_score(Y_test, model_svm.predict(X_test),average="weighted")      # 0.8489089
func_calConfusionMatrix(model_svm.predict(X_test),Y_test)

#########################
# Logistic model
#########################
accracy = []
f1 = []
for i in range(1, 10000, 100):
    LogReg = LogisticRegression(C=i)
    LogReg.fit(X_train_res, Y_train_res.values.ravel())
    accracy_score = LogReg.score(X_validation, Y_validation)
    F1 = f1_score(Y_validation, LogReg.predict(X_validation),average="weighted")
    accracy.append(accracy_score)
    f1.append(F1)

df = {'C':range(1, 10000, 100) , 'accracy':accracy, 'f1': f1}
df = pd.DataFrame(df)
print(df[df['f1'] == max(f1)])
print(df[df['accracy'] == max(accracy)])

#    C   accracy        f1
#  8101  0.675258  0.728435

# Plot the tuning process
plt.plot(df['C'], df['accracy'])
plt.title('Logistic model over validation set')
plt.xlabel('C')
plt.ylabel('accracy')
plt.show()

plt.plot(df['C'], df['f1'])
plt.title('Logistic model over validation set')
plt.xlabel('C')
plt.ylabel('F1 score')
plt.show()

# Test results
LogReg = LogisticRegression(C=8101)
LogReg.fit(X_train_res, Y_train_res.values.ravel())
LogReg.score(X_test, Y_test)                                     # 0.6567414
f1_score(Y_test, LogReg.predict(X_test),average="weighted")      # 0.7235428
func_calConfusionMatrix(LogReg.predict(X_test),Y_test)


#########################
# Random Forest (no need of normalization)
#########################
# tuning one parameter
accracy = []
f1 = []
for i in np.arange(1,201,10):
    clf_rf = RandomForestClassifier(n_estimators=i, random_state=1)
    clf_rf.fit(X_train_res, Y_train_res.values.ravel())
    accracy_score = clf_rf.score(X_validation, Y_validation)
    F1 = f1_score(Y_validation, clf_rf.predict(X_validation),average="weighted")
    accracy.append(accracy_score)
    f1.append(F1)

df = {'n_estimators':np.arange(1,201,10) , 'accracy':accracy, 'f1': f1}
df = pd.DataFrame(df)
print(df[df['f1'] == max(f1)])
print(df[df['accracy'] == max(accracy)])

#  n_estimators   accracy       f1
#      121       0.807216    0.817474
#      131       0.807216    0.817474

# Plot the tuning process
plt.plot(df['n_estimators'], df['accracy'])
plt.title('Logistic model over validation set')
plt.xlabel('n_estimators')
plt.ylabel('accracy')
plt.show()

plt.plot(df['n_estimators'], df['f1'])
plt.title('Logistic model over validation set')
plt.xlabel('n_estimators')
plt.ylabel('F1 score')
plt.show()

# tuning two parameters
accracy = []
f1 = []
n_estimators=[]
max_depth=[]
for i in np.arange(1,150,10):
    for j in np.arange(1,30):
        clf_rf = RandomForestClassifier(n_estimators=i,max_depth=j, random_state=1)
        clf_rf.fit(X_train_res, Y_train_res.values.ravel())
        accracy_score = clf_rf.score(X_validation, Y_validation)
        F1 = f1_score(Y_validation, clf_rf.predict(X_validation),average="weighted")
        accracy.append(accracy_score)
        f1.append(F1)
        n_estimators.append(i)
        max_depth.append(j)

df = {'n_estimators':n_estimators ,'max_depth':max_depth , 'accracy':accracy, 'f1': f1}
df = pd.DataFrame(df)
print(df[df['f1'] == max(f1)])
# n_estimators  max_depth   accracy        f1
#      111         29      0.808763    0.820883

print(df[df['accracy'] == max(accracy)])
#   n_estimators  max_depth   accracy        f1
#        111         29      0.808763  0.820883
#        121         29      0.808763  0.820439
#        141         29      0.808763  0.819986

# Test results
clf_rf = RandomForestClassifier(n_estimators=111,max_depth=29, random_state=1)
clf_rf.fit(X_train_res, Y_train_res.values.ravel())
clf_rf.score(X_test, Y_test)              # 0.814385
f1_score(Y_test, clf_rf.predict(X_test),average="weighted")  # 0.322033
func_calConfusionMatrix(clf_rf.predict(X_test),Y_test) 

