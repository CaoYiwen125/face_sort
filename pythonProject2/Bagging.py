#数据
from sklearn import datasets
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target
#单个分类器
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

#单个分类器
clf_knn = KNeighborsClassifier()
scores = cross_val_score(clf_knn, X, y, cv=5, scoring='roc_auc')
print('单个分类器AUC',scores.mean())
# Bagging
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(KNeighborsClassifier(),n_estimators=20, max_samples=0.5, max_features=0.5)
scores = cross_val_score(bagging, X, y, cv=5, scoring='roc_auc')
print('BaggingAUC：',scores.mean())
#__________________________________________________________________
#单个决策树
from sklearn import tree

#单个分类器
clf_tree = tree.DecisionTreeClassifier(criterion="entropy")
scores = cross_val_score(clf_tree, X, y, cv=5, scoring='roc_auc')
print('单个决策树 AUC',scores.mean())

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=20)
scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print('RandomForest AUC：',scores.mean())

######################################################Adaboost
#单个SVM

from sklearn.svm import SVC

#单个分类器
clf_svc = SVC(kernel='rbf', probability=True)
scores = cross_val_score(clf_svc, X, y, cv=5, scoring='roc_auc')
print('单个SVM AUC',scores.mean())
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

base_clf = SVC(kernel='rbf', probability=True)

adaboost = AdaBoostClassifier(estimator=base_clf, n_estimators=10)
scores = cross_val_score(adaboost, X, y, cv=5, scoring='roc_auc')
print('Adaboost-SVM AUC：',scores.mean())

################################################Stacking
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import StackingClassifier


clf1 = KNeighborsClassifier()
clf2 = SVC(kernel='linear')
clf3 = GaussianNB()
clf4 = DecisionTreeClassifier()
lr = LogisticRegression()
stacking = StackingClassifier(estimators=[('KNN',clf1), ('SVC',clf2), ('NB',clf3), ('DT',clf4)],  final_estimator=lr)

for clf, label in zip([clf1,clf2,clf3,clf4,stacking], ['KNN','SVC','Naive Bayes','Decision Tree','StackingClassifier']):
       scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
       print("AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))