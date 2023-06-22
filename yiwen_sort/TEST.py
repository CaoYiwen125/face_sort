from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc
from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import optuna
import torch
from imblearn.over_sampling import SMOTE
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


###########获取预处理后的测试数据##########
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
lda_race = LDA(n_components=5)
lda_race.fit(xtrain,ytrain[:,3])
#将训练好的LDA模型对测试数据进行特征提取
xtest_race = lda_race.transform(xtest)
xtrain_race = lda_race.transform(xtrain)
print('用训练后的种族分类LDA模型降维后的图像测试数据：',xtest_race.shape)

#创建表情分类的LDA模型降维并训练
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_face = LDA(n_components=1)
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
y_train =ytrain[:,3]
y_test =ytest[:,2]
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
    print('l',l)
    y_train[l]=y_temall[l]
    y_test[l]=y_temall[l+1996]
    l+=1
print('y_train:')
print(y_train)
print('y_test:')
print(y_test)

# y_test = le.fit_transform(y_test)
# y_train = le.fit_transform(y_train)
# print('y_test',y_test)
def random_under_sampling(X, y):
    # y_0 = np.sum(y == 0)
    # y_1 = np.sum(y == 1)
    # y_2 = np.sum(y == 2)
    # y_3 = np.sum(y == 3)
    # y_4 = np.sum(y == 4)
    # y_5 = np.sum(y == 5)
    # print(y_0,y_1,y_2,y_3,y_4,y_5)
    model_smote = SMOTE(random_state=42, k_neighbors=1)
    x_smote_resampled, y_smote_resampled = model_smote.fit_resample(X, y)  # 输入数据做过抽样处理
    # x_smote_resampled = pd.DataFrame(x_smote_resampled,columns=['col1', 'col2', 'col3', 'col4', 'col5','col6'])  # 将数据转换为数据框并命名列名
    # y_smote_resampled = pd.DataFrame(y_smote_resampled, columns=['label'])  # 将数据转换为数据框并命名列名
    # smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled], axis=1)  # 按列合并数据框
    # groupby_data_smote = smote_resampled.groupby('label').count()  # 对label做分类汇总
    # print(groupby_data_smote)
    print(x_smote_resampled,y_smote_resampled)
    return x_smote_resampled,y_smote_resampled
    # 计算需要删除的多数类样本数量
    # counter = Counter(y)
    # target_count = int(counter[1] / ratio)
    # print(target_count)
    # 获取多数类样本的索引
    # index = np.arange(len(X))[y == 1]

    # 随机选择需要删除的多数类样本
    # random_index = np.random.choice(index, size=target_count, replace=False)
    # 将需要删除的样本从数据集中删除
    # delete_index = np.concatenate([random_index, np.arange(len(X))[y == 1]])
    # X_resampled = np.delete(X, delete_index, axis=0)
    # y_resampled = np.delete(y, delete_index, axis=0)
    # return X_resampled, y_resampled

# X_resampled, y_resampled = random_under_sampling(x, y)
x_test,y_test=random_under_sampling(x_test,y_test)
x_train,y_train = random_under_sampling(x_train,y_train)
print(x_train.shape)
print(x_test.shape)