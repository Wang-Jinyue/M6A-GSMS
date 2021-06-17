##部分代码
##参考即可
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import StackingCVClassifier
import matplotlib.pyplot as plt
from sklearn import  model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import pandas as pd

datafile1 = u'F:\\N6-methyladenosine sites\\m6A\\Fusion\\GBDT\\Arabidopsis thaliana.csv'
dataset1 = pd.read_csv(datafile1,header = None)
X1 = dataset1.values[:, 0:287]
y1 = dataset1.values[:, 288]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.2)

A_RF = RandomForestClassifier(n_estimators=101, max_depth=14, max_features=16, min_samples_leaf=1, min_samples_split=2, criterion="gini")
A_ET = ExtraTreesClassifier(max_depth=19, min_samples_split=10, criterion="entropy")
A_SVM = SVC(probability=True)
A_LGBM = LGBMClassifier(max_depth=10, learning_rate=0.12, feature_fraction=0.3, lambda_l1=0.6, lambda_l2=26, cat_smooth=15)
A_Bagging = BaggingClassifier(n_estimators=141)
A_Ada = AdaBoostClassifier(learning_rate=0.4, n_estimators=98)
A_GNB = GaussianNB()
sclf = StackingCVClassifier( classifiers = [A_RF, A_ET, A_SVM, A_LGBM, A_Bagging,A_Ada],
                             use_probas = True,
                             meta_classifier = A_GNB,
                             random_state=42)

print('10-fold cross validation:\n')
A_RF.fit(X1_train, y1_train)
A_ET.fit(X1_train, y1_train)
A_SVM.fit(X1_train, y1_train)
A_LGBM.fit(X1_train, y1_train)
A_Bagging.fit(X1_train, y1_train)
A_Ada.fit(X1_train, y1_train)
A_GNB.fit(X1_train, y1_train)
sclf.fit(X1_train, y1_train)
pred1 = model_selection.cross_val_predict(sclf, X1_test, y1_test, cv=10)
print(confusion_matrix(y1_test, pred1))
pred2 = model_selection.cross_val_predict(A_RF, X1_test, y1_test, cv=10)
print(confusion_matrix(y1_test, pred2))
pred3 = model_selection.cross_val_predict(A_ET, X1_test, y1_test, cv=10)
print(confusion_matrix(y1_test, pred3))
pred4 = model_selection.cross_val_predict(A_SVM, X1_test, y1_test, cv=10)
print(confusion_matrix(y1_test, pred4))
pred5 = model_selection.cross_val_predict(A_LGBM, X1_test, y1_test, cv=10)
print(confusion_matrix(y1_test, pred5))
pred6 = model_selection.cross_val_predict(A_Bagging, X1_test, y1_test, cv=10)
print(confusion_matrix(y1_test, pred6))
pred7 = model_selection.cross_val_predict(A_Ada, X1_test, y1_test, cv=10)
print(confusion_matrix(y1_test, pred7))
pred8 = model_selection.cross_val_predict(A_GNB, X1_test, y1_test, cv=10)
print(confusion_matrix(y1_test, pred8))

##....