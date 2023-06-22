##虽然LDA模型更多用于特征提取，但本身可以用于分类，这也是FLDA的特点之一。下面是LDA直接用于分类的代码
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler

##############FLDA_extract.py#############################
##########获取数字化后的图像训练数据##########
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
#print('标签训练数据ytrain:\n',ytrain,ytrain.shape)
###########获取预处理后的测试数据##########
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
###########获取标签测试数据（这里用原始数据处理,因为我没有预处理后的faceDS）##########
'''
#如果有预处理后的faceDS文件，则将虚线内的代码改用为下面的代码：
y2 = y2.split() #默认删除所有空字符，包括空格和换行符等
del y2[0]
y2 = np.array(y2)
y2.resize((1996,5))
ytest = y2
'''
f = open("faceDS","r",encoding="utf8")
y2 = f.read()
f.close()
#——————————————————————————————————————————————————————————————————————————————————
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
#print('预处理后标签测试数据ytest删了错误行行数：',delnum)
ytest = DATAS
#——————————————————————————————————————————————————————————————————————————————
#print('标签测试数据ytest:\n',ytest,ytest.shape)


##########创建LDA模型训练后直接用于测试集的分类###########
##########性别分类##########
lda_sex = LDA(n_components=1)
lda_sex.fit(xtrain,ytrain[:,1])
#训练好的LDA模型直接用于分类
lda_sex.predict(xtest)
print('LDA分类——性别分类分数',lda_sex.score(xtest,ytest[:,1]))

##########年龄分类##########
lda_age = LDA(n_components=2)
lda_age.fit(xtrain,ytrain[:,2])
lda_age.predict(xtest)
print('LDA分类——年龄分类分数',lda_age.score(xtest,ytest[:,2]))

##########种族分类##########
lda_race = LDA(n_components=6)
lda_race.fit(xtrain,ytrain[:,3])
lda_race.predict(xtest)
print('LDA分类——种族分类分数',lda_race.score(xtest,ytest[:,3]))

##########表情分类##########
lda_face = LDA(n_components=2)
lda_face.fit(xtrain,ytrain[:,4])
lda_face.predict(xtest)
print('LDA分类——表情分类分数',lda_face.score(xtest,ytest[:,4]))


#####五折交叉验证LDA在分类上的准确率#####

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