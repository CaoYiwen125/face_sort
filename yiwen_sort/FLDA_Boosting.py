from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc
from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import optuna
import torch
#前面已经获得预处理后的训练数据
##########获取数字化后的图像训练数据##########
f = open("faceR","r",encoding="utf8")
x = f.read()
f.close()
x = x.split() #默认删除所有空字符，包括空格和换行符等
x = np.array(list(float(char) for char in x)) #字符串列表转数组
x.resize((1996,100)) #转为二维数组
xtrain = x
print('图像训练数据xtrain:\n',xtrain,xtrain.shape)
xtrain = np.delete(xtrain,0,axis=1) #删除第一列：图片编号
print('\nxtrain去编号后：\n',xtrain,xtrain.shape)
#标准化图像训练数据
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
n_features = xtrain.shape[1]
n_samples = xtrain.shape[0]
print('\nxtrain标准化后：\n',xtrain,"\n特征数：",n_features,"\n样本数：",n_samples)


#########获取预处理后的标签训练数据##########
f = open("faceDR","r",encoding="utf8")
y = f.read()
f.close()
y = y.split() #默认删除所有空字符，包括空格和换行符等
#print(y)
del y[0]
#print(y)
y = np.array(y)
y.resize((1996,5))
ytrain = y
print('标签训练数据ytrain:\n',ytrain,ytrain.shape)

#性别分类
print('\n性别分类ytrain_sex:\n',ytrain[:,1],ytrain[:,1].shape)
#年龄分类
print('\n年龄分类ytrain_age:\n',ytrain[:,2],ytrain[:,2].shape)
#种族分类
print('\n种族分类ytrain_race:\n',ytrain[:,3],ytrain[:,3].shape)
#表情分类
print('\n表情分类ytrain_face:\n',ytrain[:,4],ytrain[:,4].shape)

#faceDR中错误修改(faceDS无拼写错误)
##faceDR##
for i in range(1996):
    if ytrain[i,2] == 'chil':
        ytrain[i,2] = 'child'
    elif ytrain[i,2] == 'adulte':
        ytrain[i,2] = 'adult'
    elif ytrain[i,3] == 'whit':
        ytrain[i,3] = 'white'
    elif ytrain[i,3] == 'whitee':
        ytrain[i,3] = 'white'
    elif ytrain[i,4] == 'erious':
        ytrain[i,4] = 'serious'
    elif ytrain[i,4] == 'smilin':
        ytrain[i,4] = 'smiling'
print(ytrain)
#修正后的训练标签数据ytrain


#########获取预处理后的测试数据##########
f = open("faceS","r",encoding="utf8")
x2 = f.read()
f.close()
x2 = x2.split() #默认删除所有空字符，包括空格和换行符等
x2 = np.array(list(float(char) for char in x2)) #字符串列表转数组
x2.resize((1996,100)) #转为二维数组
xtest = x2
print('图像测试数据xtest:\n',xtest,xtest.shape)
xtest = np.delete(xtest,0,axis=1) #删除第一列：图片编号
print('\nxtest去编号后：\n',xtest,xtest.shape)
xtest = scaler.fit_transform(xtest)
print('\nxtest标准化后：\n',xtest)


###########获取标签测试数据（这里用原始数据处理,因为我没有预处理后的faceDS）##########
'''
#如果有预处理后的faceDS文件，则将虚线内的代码改用为下面的代码：
y2 = y2.split() #默认删除所有空字符，包括空格和换行符等
del y2[0]
y2 = np.array(y2)
y2.resize((1996,5))
ytest = y2
'''
f = open("faceDS1.txt","r",encoding="utf8")
y2 = f.read()
f.close()

#——————————————————————————————————————————————————————————————————————————————————
'''
y2 = y2.split(' ')
#print(y2)
import copy
start_num = 3223
end_num = 5222
datanum = 2000
DATA = [['0']*5 for i in range(2000)]

num = 1
i = 0
while num <= len(y2):
    cout = start_num+i
    if cout > end_num:
        break
    cout=str(cout)
    if y2[num] == cout:
        DATA[i][0] = y2[num]
        num+=1
        if y2[num]=='(_sex':
            num+=2
            y2[num] = y2[num].replace(')','')
            DATA[i][1] = y2[num]
            num+=3
            y2[num] = y2[num].replace(')','')
            DATA[i][2] = y2[num]
            num+=2
            y2[num] = y2[num].replace(')','')
            DATA[i][3] = y2[num]
            num+=2
            y2[num] = y2[num].replace(')','')
            DATA[i][4] = y2[num]
            num+=2
            num += 1
            i += 1
        else:
            num+=1
            i += 1
    else:
        num+=1
DATAS = copy.deepcopy(DATA)
DATAS = np.array(DATAS)
#print(DATAS)
matrix = DATAS[:,0]
#print(matrix)
long = len(matrix)
#print(long)
j = 0
h = 0
delnum = 0
while h < long:
    if DATAS[j,1]=='0':
        DATAS = np.delete(DATAS,j,axis=0)
        #print('删了',h+3223)
        j-=1
        delnum+=1
    j+=1
    h+=1
print('预处理后标签测试数据ytest删了错误行行数：',delnum)
ytest = DATAS
'''
#——————————————————————————————————————————————————————————————————————————————
y2 = y2.split() #默认删除所有空字符，包括空格和换行符等
del y2[0]
y2 = np.array(y2)
y2.resize((1996,5))
ytest = y2
#=============================================================

print('标签测试数据ytest:\n',ytest,ytest.shape)
#性别分类
print('\n性别分类ytest_sex:\n',ytest[:,0],ytest[:,0].shape)
#年龄分类
print('\n年龄分类ytest_age:\n',ytest[:,1],ytest[:,1].shape)
#种族分类
print('\n种族分类ytest_race:\n',ytest[:,2],ytest[:,2].shape)
#表情分类
print('\n表情分类ytest_face:\n',ytest[:,3],ytest[:,3].shape)




##########创建LDA模型降维并训练###########
##不同于PCA用于无监督学习的特征提取，FLDA适用于监督学习的特征提取，即需要标签y一起送入LDA模型中训练
#创建性别分类的LDA模型降维并训练
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_sex = LDA(n_components=1) #监督学习下的FLDA特征提取，数据降维后的维数受特征维数和类别数的限制；PCA较自由
lda_sex.fit(xtrain,ytrain[:,1])
#将训练好的LDA模型对测试数据进行特征提取
xtest_sex = lda_sex.transform(xtest)
xtrain_sex = lda_sex.transform(xtrain)
print('用训练后的性别分类LDA模型降维后的图像测试数据：',xtest_sex.shape)

#创建年龄分类的LDA模型降维并训练
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_age = LDA(n_components=3)
lda_age.fit(xtrain,ytrain[:,2])
#将训练好的LDA模型对测试数据进行特征提取
xtest_age = lda_age.transform(xtest)
xtrain_age = lda_age.transform(xtrain)
print('用训练后的年龄分类LDA模型降维后的图像测试数据：',xtest_age.shape)

#创建种族分类的LDA模型降维并训练
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_race = LDA(n_components=4)
lda_race.fit(xtrain,ytrain[:,3])
#将训练好的LDA模型对测试数据进行特征提取
xtest_race = lda_race.transform(xtest)
xtrain_race = lda_race.transform(xtrain)
print('用训练后的种族分类LDA模型降维后的图像测试数据：',xtest_race.shape)

#创建表情分类的LDA模型降维并训练
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_face = LDA(n_components=2)
lda_face.fit(xtrain,ytrain[:,4])
#将训练好的LDA模型对测试数据进行特征提取
xtest_face = lda_face.transform(xtest)
xtrain_face = lda_face.transform(xtrain)
print('用训练后的表情分类LDA模型降维后的图像测试数据：',xtest_face.shape)

# ----------------------------------------------------------------------------------------------
# 进行数据分析，使用随机森林来训练和预测数据集
# 处理数据，找出目标值和特征值
# x = datas[['pclass', 'age', 'sex', 'embarked']]
# y = datas['survived']

# 先分割数据集为训练集和测试集
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)#选择的比例
# print(x_train.head(1))
#sex\age\race\face
x_train = xtrain_age
x_test = xtest_age
y_train =ytrain[:,2]
y_test =ytest[:,1]
print(y_test)
# 进行处理（特征工程）—— 类别（one-hot编码）
# 进行特征工程，pd转换字典，特征抽取  x_train.to_dict(orient="records")
# dv = DictVectorizer(sparse=False)
# x_train = dv.fit_transform(x_train.to_dict(orient='records'))
# x_test = dv.transform(x_test.to_dict(orient='records'))
##数据race等不平衡处理
##独热编码
# y_train=pd.get_dummies(y_train)
# y_test=pd.get_dummies(y_test)
# print('y_test_one_hot:')
# print(y_test)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# y_temall = np.vstack(y_train,y_test)
# y_temall = le.fit_transform(y_temall)
y_temall = np.append(y_train,y_test,axis=0)
y_temall = le.fit_transform(y_temall)
print(y_temall,len(y_temall))
l = 0
while l<=1995:
    # print('l',l)
    y_train[l]=y_temall[l]
    y_test[l]=y_temall[l+1996]
    l+=1
# y_train = list(y_train)
# y_test = list(y_test)
# y_train_spe = label_binarize(y_train,classes=['0','1','2','3'])


print('y_train:')
print(y_train)
print('y_test:')
print(y_test)
print(np.sum(y_train=='0'))
# y_test = le.fit_transform(y_test)
# y_train = le.fit_transform(y_train)
# print('y_test',y_test)

def random_under_sampling(X, y, ratio=0.5):
    # 计算需要删除的多数类样本数量
    # counter = Counter(y)
    # target_count = int(counter[1] / ratio)
    # 获取多数类样本的索引
    # index = np.arange(len(X))[y == 1]
    # 随机选择需要删除的多数类样本
    # random_index = np.random.choice(index, size=target_count, replace=True)
    # 将需要删除的样本从数据集中删除
    # delete_index = np.concatenate([random_index, np.arange(len(X))[y == 1]])
    # X_resampled = np.delete(X, delete_index, axis=0)
    # y_resampled = np.delete(y, delete_index, axis=0)
    # return X_resampled, y_resampled
    model_smote = SMOTE(random_state=42, k_neighbors=5)
    x_smote_resampled, y_smote_resampled = model_smote.fit_resample(X, y)  # 输入数据做过抽样处理
    print(x_smote_resampled, y_smote_resampled)
    return x_smote_resampled, y_smote_resampled

# X_resampled, y_resampled = random_under_sampling(x, y)

# x_test,y_test=random_under_sampling(x_test,y_test)
x_train,y_train = random_under_sampling(x_train,y_train)#####过拟合处理
print(x_train.shape)
print(np.sum(y_train=='0'))
print(x_test.shape)
# y_train_pre = preprocessing.label_binarize(y_train, classes=['0','1','2','3'])

# 利用随机树进行预测
# rf = RandomForestClassifier()
# rf = SVC(kernel='rbf',probability=True)
'''
#参数循环
QianNum = 0
HouNum = 40
MidNum = 20
i = 0
k = 0
scores = []
maxscNum= 1
while i < 4:
    j = 0
    for k in range(QianNum,HouNum):
    # while k<(HouNum-QianNum):
        print(QianNum, HouNum, 'is start',k)
        rfc = RandomForestClassifier(n_estimators=QianNum+k+1)
        score1 = cross_val_score(rfc,x_train,y_train,cv=5).mean() #交叉验证得平均分
        scores.append(score1)
        if scores[j] == max(scores):
            maxscNum = k
        print(max(scores),maxscNum) #打印最高分数，及其索引
        j+=1
    plt.figure(figsize=[20,5])
    print(QianNum,HouNum,'is get pltplot')
    plt.plot(range(QianNum,HouNum),scores)
    plt.show()
    if i<3 :
        if maxscNum<=(HouNum+QianNum)/2:
            QianNum = QianNum
            HouNum = int((HouNum + QianNum) / 2)
            print(i,QianNum,HouNum)
        else:
            QianNum = int((HouNum + QianNum) / 2)
            HouNum = HouNum
            print(i, QianNum, HouNum)
    scores = []
    i+=1
print('n_estimators 已选定范围')
# if QianNum == 0:
#     QianNum+=1
i = 1
Depth = []
param_grid = {'max_depth':np.arange(1, 30, 1)}
while i <(HouNum-QianNum+1):
    rfc = RandomForestClassifier(n_estimators=QianNum+i)
    # 网格搜索模型，参数1：随机森林模型；参数2：待调整的参数；参数3：交叉验证的次数
    GS1 = GridSearchCV(rfc,param_grid,cv=5)
    GS1.fit(x_train,y_train)
    Depth.append(GS1.best_params_.get('max_depth'))
    print(str(QianNum+i)+"最佳参数:")
    print(GS1.best_params_)
    print(str(QianNum+i)+"参数得分:")
    print(GS1.best_score_)
    i+=1
'''
####调整参数n_estimator————————————
'''
scores = []
for i in range(10,20):
    rfc = RandomForestClassifier(n_estimators=i+1)
    score1 = cross_val_score(rfc,x_train,y_train,cv=5).mean() #交叉验证得平均分
    scores .append(score1)
    print(max(scores ),(scores .index(max(scores))*10)+1) #打印最高分数，及其索引
plt.figure(figsize=[20,5])
plt.plot(range(10,20),scores)
plt.show()
'''
####调整参数max_depth————————————
####网格搜索：按照参数对模型整体准确率的影响程度进行调参
'''
param_grid = {'max_depth':np.arange(1, 30, 1)}
rfc = RandomForestClassifier(n_estimators=20)
# 网格搜索模型，参数1：随机森林模型；参数2：待调整的参数；参数3：交叉验证的次数
GS1 = GridSearchCV(rfc,param_grid,cv=5)
GS1.fit(x_train,y_train)
print("最佳参数:")
print(GS1.best_params_)
print("参数得分:")
print(GS1.best_score_)

k = 0
N_ES=[]
# while k<9:
while k < 5:
    N_ES.append(int(QianNum+k+1))
    k+=1
    # N_ES=int(N_ES)
print('N-ES',N_ES)
Depth = list(set(Depth))
print('Depth',Depth)


# 构建超参数字典
param_dict = {'n_estimators':N_ES,'max_depth':Depth}
'''
import time
start_time = time.time()    # 程序开始时间
# score = np.zeros(10)
# i=0
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import  classification_report,confusion_matrix
from sklearn.model_selection import  train_test_split
# '''
# ascore = []
# m = 10
# while m<=81:
#     gbc = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=20, max_depth=13,
#                                      min_samples_leaf=150, min_samples_split=400,
#                                      max_features=1, subsample=0.9, random_state=10, verbose=1)#sex
#     gbc.fit(x_train, y_train)
#     ascore.append()
#     m=m+10

gbc = GradientBoostingClassifier(loss='log_loss',learning_rate=0.01,n_estimators=100,max_depth=30,
                                 min_samples_leaf =20,min_samples_split=100,
                                 max_features=15, subsample=0.9, random_state=10,verbose=1)#age

# gbc = GradientBoostingClassifier(loss='log_loss',learning_rate=0.005,n_estimators=40,max_depth=20,
#                                  min_samples_leaf =20,min_samples_split=50,
#                                  max_features=1, subsample=0.9, random_state=10,verbose=1)#race
# gbc = GradientBoostingClassifier(loss='log_loss',learning_rate=0.01,n_estimators=40,max_depth=10,
#                                  min_samples_leaf =10,min_samples_split=20,
#                                  max_features=1, subsample=0.9, random_state=10,verbose=1)#face

gbc.fit(x_train, y_train)

y_pred = gbc.predict(x_test)
y_predprob = gbc.predict_proba(x_test)[:,1]

report = classification_report(y_test, gbc.predict(x_test), output_dict=False)
print('report:',report)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, robust=True)
plt.show()
#交叉验证部分
score = cross_val_score(gbc,x_train,y_train,cv=5).mean()
print('交叉验证准确率：',score)


# '''
#设计步长0.1，从n_estimators开始调参
'''
param_test1 = {'n_estimators':range(10,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
                        param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(x_train,y_train)
print(gsearch1.cv_results_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)
'''
#max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索
'''
param_test2 = {'max_depth':range(3,24,2), 'min_samples_split':range(100,801,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, min_samples_leaf=20,
                                                               max_features='sqrt', subsample=0.8, random_state=10),
                        param_grid = param_test2, scoring='roc_auc', cv=5)
gsearch2.fit(x_train,y_train)
print(gsearch2.cv_results_)
print(gsearch2.best_params_)
print(gsearch2.best_score_)
'''
#最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf
'''
param_test3 = {'min_samples_split':range(200,1900,200), 'min_samples_leaf':range(0,400,50)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=20,max_depth=13,
                                                               max_features='sqrt', subsample=0.8, random_state=10),
                        param_grid = param_test3, scoring='roc_auc', cv=5)
gsearch3.fit(x_train,y_train)
print(gsearch3.cv_results_)
print(gsearch3.best_params_)
print(gsearch3.best_score_)
'''

#最大特征数max_features进行网格搜索。
'''
param_test4 = {'max_features':range(1,20,1)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=20,max_depth=13, min_samples_leaf =150,
                                                               min_samples_split =400, subsample=0.8, random_state=10),
                        param_grid = param_test4, scoring='roc_auc', cv=5)
gsearch4.fit(x_train,y_train)
print(gsearch4.cv_results_)
print(gsearch4.best_params_)
print(gsearch4.best_score_)
'''

#再对子采样的比例进行网格搜索：
'''
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=20,max_depth=13, min_samples_leaf =150,
                                                               min_samples_split =400, max_features=1, random_state=10),
                        param_grid = param_test5, scoring='roc_auc', cv=5)
gsearch5.fit(x_train,y_train)
print(gsearch5.cv_results_)
print(gsearch5.best_params_)
print(gsearch5.best_score_)

'''

#####________________

# report = classification_report(y_test, gbc.predict(x_test), output_dict=False)
# print('report:',report)

# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, robust=True)
# plt.show()
#交叉验证部分
# score = cross_val_score(gbc,x_train,y_train,cv=5).mean()
# print('交叉验证准确率：',score)

end_time = time.time()    # 程序结束时间

#求ROC、AUC(若不是二分类模型，注释掉)
# roc_auc_score(y_test,y_pred)
# '''
#求ROC、AUC
## 可视化算法的ROC曲线
y_test_1 = deepcopy(y_test)
classes = list(set(y_test_1))
print(y_test_1)
classes1 = str(classes)
print(classes1)
#
y_test_lb = label_binarize(y_test_1,classes=['0','1','2','3'])
n_classes = y_test_lb.shape[1]
print(n_classes)
y_score = gbc.predict_proba(x_test)
# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_lb[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 汇总所以fpr(拼接去除重复元素)后续再线性插值
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
#取平均
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# Plot all ROC curves
lw=2
plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
# '''



run_time = end_time - start_time    # 程序的运行时间，单位为秒
print('所需时间：',run_time/10)