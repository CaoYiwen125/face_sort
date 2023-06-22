##########bayes分类##########
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib
import matplotlib.pyplot as plt

############## 一、FLDA_extract.py #############################
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


########### 3、获取预处理后的测试数据 ##########
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
#del y2[0]  #faceDS文件比faceDR文件少了第一个多余的字符串，所以不用del掉。建议先print测试一下
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
print('用训练后的性别分类LDA模型降维后的图像训练数据和测试数据：',xtrain_sex.shape,xtest_sex.shape))

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
#————————————————————————————————————————————————————————————————————————————————————————————————

############### 二、送入Bayes分类 ########################
#####性别分类#####
bys_sex = GaussianNB().fit(xtrain_sex, ytrain[:,1]) #默认alpha=1.0，拉普拉斯平滑
bys_sex.predict(xtest_sex)
print('BAYES分类——性别分类分数',bys_sex.score(xtest_sex,ytest[:,1]))
#####年龄分类#####
bys_age = GaussianNB().fit(xtrain_age, ytrain[:,2]) #默认alpha=1.0，拉普拉斯平滑
bys_age.predict(xtest_age)
print('BAYES分类——年龄分类分数',bys_age.score(xtest_age,ytest[:,2]))
#####种族分类#####
bys_race = GaussianNB().fit(xtrain_race, ytrain[:,3]) #默认alpha=1.0，拉普拉斯平滑
bys_race.predict(xtest_race)
print('BAYES分类——种族分类分数',bys_race.score(xtest_race,ytest[:,3]))
#####表情分类#####
bys_face = GaussianNB().fit(xtrain_face, ytrain[:,4]) #默认alpha=1.0，拉普拉斯平滑
bys_face.predict(xtest_face)
print('BAYES分类——表情分类分数',bys_face.score(xtest_face,ytest[:,4]))


##### 三、五折交叉验证Bayes在分类上的准确率 #####
#合并x数据和y数据
X = np.concatenate([xtrain,xtest],axis=0)
Y = np.concatenate([ytrain,ytest],axis=0)
print('所有图像数据：\n',X,X.shape,'\n所有标签数据：\n',Y,Y.shape)

kfold = KFold(n_splits=5,shuffle=True,random_state=42)
sex_result = cross_val_score(bys_sex,X,Y[:,1],cv=kfold)
print('性别BAYES分类交叉验证结果：%0.2f(±%0.2f)'%(sex_result.mean(),sex_result.std()*2))
age_result = cross_val_score(bys_age,X,Y[:,2],cv=kfold)
print('年龄BAYES分类交叉验证结果：%0.2f(±%0.2f)'%(age_result.mean(),age_result.std()*2))
race_result = cross_val_score(bys_race,X,Y[:,3],cv=kfold)
print('种族BAYES分类交叉验证结果：%0.2f(±%0.2f)'%(race_result.mean(),race_result.std()*2))
face_result = cross_val_score(bys_face,X,Y[:,4],cv=kfold)
print('表情BAYES分类交叉验证结果：%0.2f(±%0.2f)'%(face_result.mean(),face_result.std()*2))


############ 四、绘制分数对比图 #####################
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
plt.rcParams['figure.figsize'] = (9.0, 5.5)
fig,ax = plt.subplots()
#绘制Bar
labels = ['sex',"age",'race','face']
num = np.arange(4)
result1 = np.array([bys_sex.score(xtest_sex,ytest[:,1]),bys_age.score(xtest_age,ytest[:,2]),bys_race.score(xtest_race,ytest[:,3]),bys_face.score(xtest_face,ytest[:,4])])
result2 = np.array([sex_result.mean(),age_result.mean(),race_result.mean(),face_result.mean()])
rects1 = ax.bar(num-0.3/2,result1,width = 0.3,label = 'predict_score')
rects2 = ax.bar(num+0.3/2,result2,width=0.3,label = 'cross_val_score')

ax.set_ylabel("scores")
ax.set_title("FLDA_Bayes:Contrast of scores")
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
plt.savefig("FLDA_Bayes.jpg")
plt.show()