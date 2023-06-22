import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib
import matplotlib.pyplot as plt

#一、特征提取
########## 1、获取数字化后的图像训练数据 ##########
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
######### 2、获取预处理后的标签训练数据 ##########
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
#faceDR中错误修改
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
print('标签训练数据ytrain:\n',ytrain,ytrain.shape)

#性别分类
print('\n性别分类ytrain_sex:\n',ytrain[:,1],ytrain[:,1].shape)
#年龄分类
print('\n年龄分类ytrain_age:\n',ytrain[:,2],ytrain[:,2].shape)
#种族分类
print('\n种族分类ytrain_race:\n',ytrain[:,3],ytrain[:,3].shape)
#表情分类
print('\n表情分类ytrain_face:\n',ytrain[:,4],ytrain[:,4].shape)
########### 3、获取预处理后的测试数据 #########
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
########### 4、获取标签测试数据 ##########
f = open("faceDS","r",encoding="utf8")
y2 = f.read()
f.close()
y2 = y2.split() #默认删除所有空字符，包括空格和换行符等
#del y2[0]  #faceDS文件比faceDR文件少了第一个多余的字符串，所以不用del掉。建议此处先print测试一下
y2 = np.array(y2)
y2.resize((1996,5))
ytest = y2
print('标签测试数据ytest:\n',ytest,ytest.shape)
#性别分类
print('\n性别分类ytest_sex:\n',ytest[:,1],ytest[:,1].shape)
#年龄分类
print('\n年龄分类ytest_age:\n',ytest[:,2],ytest[:,2].shape)
#种族分类
print('\n种族分类ytest_race:\n',ytest[:,3],ytest[:,3].shape)
#表情分类
print('\n表情分类ytest_face:\n',ytest[:,4],ytest[:,4].shape)
########## 5、创建LDA模型降维并训练 ###########
##不同于PCA用于无监督学习的特征提取，FLDA适用于监督学习的特征提取，即需要标签y一起送入LDA模型中训练
#创建性别分类的LDA模型降维并训练
lda_sex = LDA(n_components=1) #监督学习下的FLDA特征提取，数据降维后的维数受特征维数和类别数的限制，不能超过（类别数-1）；PCA较自由
lda_sex.fit(xtrain,ytrain[:,1]) #最大不能超过2-1=1
#将训练好的LDA模型对数据进行特征提取
xtrain_sex = lda_sex.transform(xtrain)
xtest_sex = lda_sex.transform(xtest)
print('用训练后的性别分类LDA模型降维后的图像训练数据和测试数据：',xtrain_sex.shape,xtest_sex.shape)

#创建年龄分类的LDA模型降维并训练
lda_age = LDA(n_components=3) #最大不能超过4-1=3
lda_age.fit(xtrain,ytrain[:,2])
#将训练好的LDA模型对数据进行特征提取
xtrain_age = lda_age.transform(xtrain)
xtest_age = lda_age.transform(xtest)
print('用训练后的年龄分类LDA模型降维后的图像训练数据和测试数据：',xtrain_age.shape,xtest_age.shape)

#创建种族分类的LDA模型降维并训练
lda_race = LDA(n_components=4) #最大不能超过5-1=4
lda_race.fit(xtrain,ytrain[:,3])
#将训练好的LDA模型对数据进行特征提取
xtrain_race = lda_race.transform(xtrain)
xtest_race = lda_race.transform(xtest)
print('用训练后的种族分类LDA模型降维后的图像训练数据和测试数据：',xtrain_race.shape,xtest_race.shape)

#创建表情分类的LDA模型降维并训练
lda_face = LDA(n_components=2) #最大不能超过3-1=2
lda_face.fit(xtrain,ytrain[:,4])
#将训练好的LDA模型对数据进行特征提取
xtrain_face = lda_face.transform(xtrain)
xtest_face = lda_face.transform(xtest)
print('用训练后的表情分类LDA模型降维后的图像训练数据和测试数据：',xtrain_face.shape,xtest_face.shape)


#二、SVM分类
###########################注释掉的内容为调参代码
# #标签数据预处理  
# le = LabelEncoder()  
# ytrain_sex = le.fit_transform(ytrain[:,1]) 
# ytest_sex = le.transform(ytest[:,1])
# #用GridSearch调参
# param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,0.1, 1, 10],
#                      'C': [0.1,1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [0.1,1, 10, 100, 1000]}]
# grid_search = GridSearchCV(SVC(), param_grid, cv=5)
# ##性别分类
# grid_search.fit(xtrain_sex, ytrain_sex)
# print("模型最优参数: ", grid_search.best_params_)
# print("准确率: ", grid_search.best_score_)
svm_sex = SVC(kernel='rbf',C=10,gamma=0.001) #创建最优参数的分类器并训练
svm_sex.fit(xtrain_sex, ytrain_sex)
predict_sex = np.array(le.inverse_transform(svm_sex.predict(xtest_sex))) #预测
print('SVM分类——性别分类预测结果：',predict_sex,predict_sex.shape)
print('SVM分类——性别分类分数：',svm_sex.score(xtest_sex,ytest_sex))

# #标签数据预处理   
# ytrain_age = le.fit_transform(ytrain[:,2]) 
# ytest_age = le.transform(ytest[:,2])
# ##年龄分类
# grid_search.fit(xtrain_age, ytrain_age)
# print("模型最优参数: ", grid_search.best_params_)
# print("准确率: ", grid_search.best_score_)
svm_age = SVC(kernel='rbf',C=10,gamma=0.1) #创建最优参数的分类器并训练
svm_age.fit(xtrain_age, ytrain_age)
predict_age = np.array(le.inverse_transform(svm_age.predict(xtest_age))) #预测
print('SVM分类——年龄分类预测结果：',predict_age,predict_age.shape)
print('SVM分类——年龄分类分数：',svm_age.score(xtest_age,ytest_age))

# #标签数据预处理   
# ytrain_race = le.fit_transform(ytrain[:,3]) 
# ytest_race = le.transform(ytest[:,3])
# ##年龄分类
# grid_search.fit(xtrain_race, ytrain_race)
# print("模型最优参数: ", grid_search.best_params_)
# print("准确率: ", grid_search.best_score_)
svm_race = SVC(kernel='rbf',C=1,gamma=0.1) #创建最优参数的分类器并训练
svm_race.fit(xtrain_race, ytrain_race)
predict_race = np.array(le.inverse_transform(svm_race.predict(xtest_race))) #预测
print('SVM分类——种族分类预测结果：',predict_race,predict_race.shape)
print('SVM分类——种族分类分数：',svm_race.score(xtest_race,ytest_race))

# #标签数据预处理   
# ytrain_face = le.fit_transform(ytrain[:,4]) 
# ytest_face = le.transform(ytest[:,4])
# ##年龄分类
# grid_search.fit(xtrain_face, ytrain_face)
# print("模型最优参数: ", grid_search.best_params_)
# print("准确率: ", grid_search.best_score_)
svm_face = SVC(kernel='linear',C=1) #创建最优参数的分类器并训练
svm_face.fit(xtrain_face, ytrain_face)
predict_face = np.array(le.inverse_transform(svm_face.predict(xtest_face))) #预测
print('SVM分类——表情分类预测结果：',predict_face,predict_face.shape)
print('SVM分类——表情分类分数：',svm_face.score(xtest_face,ytest_face))


#三、五折交叉验证
#合并x数据和y数据
X = np.concatenate([xtrain,xtest],axis=0)
Y = np.concatenate([ytrain,ytest],axis=0)
print('所有图像数据：\n',X,X.shape,'\n所有标签数据：\n',Y,Y.shape)
#交叉验证
kfold = KFold(n_splits=5,shuffle=True,random_state=42)

Y_sex = le.fit_transform(Y[:,1])
sex_result = cross_val_score(svm_sex,X,Y_sex,cv=kfold)
print('性别SVM分类交叉验证结果：%0.2f(±%0.2f)'%(sex_result.mean(),sex_result.std()*2))

Y_age = le.fit_transform(Y[:,2])
age_result = cross_val_score(svm_age,X,Y_age,cv=kfold)
print('年龄SVM分类交叉验证结果：%0.2f(±%0.2f)'%(age_result.mean(),age_result.std()*2))

Y_race = le.fit_transform(Y[:,3])
race_result = cross_val_score(svm_race,X,Y_race,cv=kfold)
print('种族SVM分类交叉验证结果：%0.2f(±%0.2f)'%(race_result.mean(),race_result.std()*2))

Y_face = le.fit_transform(Y[:,4])
face_result = cross_val_score(svm_face,X,Y_face,cv=kfold)
print('表情SVM分类交叉验证结果：%0.2f(±%0.2f)'%(face_result.mean(),face_result.std()*2))


#四、绘制分数对比图
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
plt.rcParams['figure.figsize'] = (9.0, 5.5)
fig,ax = plt.subplots()
#绘制Bar
labels = ['sex',"age",'race','face']
num = np.arange(4)
result1 = np.array([svm_sex.score(xtest_sex,ytest_sex),svm_age.score(xtest_age,ytest_age),svm_race.score(xtest_race,ytest_race),svm_face.score(xtest_face,ytest_face)])
result2 = np.array([sex_result.mean(),age_result.mean(),race_result.mean(),face_result.mean()])
rects1 = ax.bar(num-0.3/2,result1,width = 0.3,label = 'predict_score')
rects2 = ax.bar(num+0.3/2,result2,width=0.3,label = 'cross_val_score')

ax.set_ylabel("scores")
ax.set_title("FLDA_SVM:Contrast of scores")
ax.set_xticks(num)
ax.set_xticklabels(labels)
ax.legend()
#绘制标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2,height),
                   xytext=(0,3),
                   textcoords="offset points",
                   ha='center',va='bottom')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig("FLDA_SVM.jpg")
plt.show()