#数据
from sklearn import datasets
cancer = datasets.load_breast_cancer()
# print('cancer_type',type(cancer))
# print('cancer',cancer)
X, y = cancer.data, cancer.target
# print('x:',type(X),'y',type(y))
# print('X:',X)
# print('Y',y)
#单个分类器
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

#单个分类器
clf_knn = KNeighborsClassifier()
scores = cross_val_score(clf_knn, X, y, cv=5, scoring='roc_auc')
print('单个分类器AUC',scores.mean())



