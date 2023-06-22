##虽然LDA模型更多用于特征提取，但本身可以用于分类，这也是FLDA的特点之一。下面是LDA直接用于分类的代码
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler
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
#print('图像训练数据xtrain:\n',xtrain,xtrain.shape)
xtrain = np.delete(xtrain,0,axis=1) #删除第一列：图片编号
#print('\nxtrain去编号后：\n',xtrain,xtrain.shape)
#标准化图像训练数据
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
n_features = xtrain.shape[1]
n_samples = xtrain.shape[0]
#print('\nxtrain标准化后：\n',xtrain,"\n特征数：",n_features,"\n样本数：",n_samples)
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
#print('标签训练数据ytrain:\n',ytrain,ytrain.shape)
########### 3、获取预处理后的测试数据 ##########
f = open("faceS","r",encoding="utf8")
x2 = f.read()
f.close()
x2 = x2.split() #默认删除所有空字符，包括空格和换行符等
x2 = np.array(list(float(char) for char in x2)) #字符串列表转数组
x2.resize((1996,100)) #转为二维数组
xtest = x2
#print('图像测试数据xtest:\n',xtest,xtest.shape)
xtest = np.delete(xtest,0,axis=1) #删除第一列：图片编号
#print('\nxtest去编号后：\n',xtest,xtest.shape)
xtest = scaler.fit_transform(xtest)
#print('\nxtest标准化后：\n',xtest)
########### 4、获取标签测试数据 ##########
f = open("faceDS","r",encoding="utf8")
y2 = f.read()
f.close()
y2 = y2.split() #默认删除所有空字符，包括空格和换行符等
#del y2[0]  #faceDS文件比faceDR文件少了第一个多余的字符串，所以不用del掉。建议此处先print测试一下
y2 = np.array(y2)
y2.resize((1996,5))
ytest = y2
#print('标签测试数据ytest:\n',ytest,ytest.shape)
#————————————————————————————————————————————————————————————————————————————————————————————————

########## 二、创建LDA模型训练后直接用于测试集的分类 ###########
##########性别分类##########
lda_sex = LDA(n_components=1)
lda_sex.fit(xtrain,ytrain[:,1])
predict_sex = lda_sex.predict(xtest) #训练好的LDA模型直接用于分类
print('LDA分类——性别分类预测结果：',predict_sex,predict_sex.shape)
print('LDA分类——性别分类分数：',lda_sex.score(xtest,ytest[:,1]))

##########年龄分类##########
lda_age = LDA(n_components=3)
lda_age.fit(xtrain,ytrain[:,2])
predict_age = lda_age.predict(xtest)
print('LDA分类——年龄分类预测结果：',predict_age,predict_age.shape)
print('LDA分类——年龄分类分数：',lda_age.score(xtest,ytest[:,2]))

##########种族分类##########
lda_race = LDA(n_components=4)
lda_race.fit(xtrain,ytrain[:,3])
predict_race = lda_race.predict(xtest)
print('LDA分类——种族分类预测结果：',predict_race,predict_race.shape)
print('LDA分类——种族分类分数：',lda_race.score(xtest,ytest[:,3]))

##########表情分类##########
lda_face = LDA(n_components=2)
lda_face.fit(xtrain,ytrain[:,4])
predict_face = lda_face.predict(xtest)
print('LDA分类——表情分类预测结果：',predict_face,predict_face.shape)
print('LDA分类——表情分类分数',lda_face.score(xtest,ytest[:,4]))


##### 三、五折交叉验证LDA在分类上的准确率 #####
#合并x数据和y数据
X = np.concatenate([xtrain,xtest],axis=0)
Y = np.concatenate([ytrain,ytest],axis=0)
print('所有图像数据：\n',X,X.shape,'\n所有标签数据：\n',Y,Y.shape)

kfold = KFold(n_splits=5,shuffle=True,random_state=42)
sex_result = cross_val_score(lda_sex,X,Y[:,1],cv=kfold)
print('性别LDA分类交叉验证结果：%0.2f(±%0.2f)'%(sex_result.mean(),sex_result.std()*2))
age_result = cross_val_score(lda_age,X,Y[:,2],cv=kfold)
print('年龄LDA分类交叉验证结果：%0.2f(±%0.2f)'%(age_result.mean(),age_result.std()*2))
race_result = cross_val_score(lda_race,X,Y[:,3],cv=kfold)
print('种族LDA分类交叉验证结果：%0.2f(±%0.2f)'%(race_result.mean(),race_result.std()*2))
face_result = cross_val_score(lda_face,X,Y[:,4],cv=kfold)
print('表情LDA分类交叉验证结果：%0.2f(±%0.2f)'%(face_result.mean(),face_result.std()*2))

############ 四、绘制分数对比图 #####################
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
plt.rcParams['figure.figsize'] = (9.0, 5.5)
fig,ax = plt.subplots()
#绘制Bar
labels = ['sex',"age",'race','face']
num = np.arange(4)
result1 = np.array([lda_sex.score(xtest,ytest[:,1]),lda_age.score(xtest,ytest[:,2]),lda_race.score(xtest,ytest[:,3]),lda_face.score(xtest,ytest[:,4])])
result2 = np.array([sex_result.mean(),age_result.mean(),race_result.mean(),face_result.mean()])
rects1 = ax.bar(num-0.3/2,result1,width = 0.3,label = 'predict_score')
rects2 = ax.bar(num+0.3/2,result2,width=0.3,label = 'cross_val_score')

ax.set_ylabel("scores")
ax.set_title("FLDA_classify:Contrast of scores")
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
plt.savefig("FLDA_classify.jpg")
plt.show()