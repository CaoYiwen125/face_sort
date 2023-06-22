import numpy as np
from sklearn.preprocessing import StandardScaler
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
print('预处理后标签测试数据ytest删了错误行行数：',delnum)
ytest = DATAS
#——————————————————————————————————————————————————————————————————————————————


print('标签测试数据ytest:\n',ytest,ytest.shape)
#性别分类
print('\n性别分类ytest_sex:\n',ytest[:,1],ytest[:,1].shape)
#年龄分类
print('\n年龄分类ytest_age:\n',ytest[:,2],ytest[:,2].shape)
#种族分类
print('\n种族分类ytest_race:\n',ytest[:,3],ytest[:,3].shape)
#表情分类
print('\n表情分类ytest_face:\n',ytest[:,4],ytest[:,4].shape)


##########创建LDA模型降维并训练###########
##不同于PCA用于无监督学习的特征提取，FLDA适用于监督学习的特征提取，即需要标签y一起送入LDA模型中训练
#创建性别分类的LDA模型降维并训练
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_sex = LDA(n_components=1) #监督学习下的FLDA特征提取，数据降维后的维数受特征维数和类别数的限制；PCA较自由
lda_sex.fit(xtrain,ytrain[:,1])
#将训练好的LDA模型对测试数据进行特征提取
xtest_sex = lda_sex.transform(xtest)
print('用训练后的性别分类LDA模型降维后的图像测试数据：',xtest_sex.shape)

#创建年龄分类的LDA模型降维并训练
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_age = LDA(n_components=2)
lda_age.fit(xtrain,ytrain[:,2])
#将训练好的LDA模型对测试数据进行特征提取
xtest_age = lda_age.transform(xtest)
print('用训练后的年龄分类LDA模型降维后的图像测试数据：',xtest_age.shape)

#创建种族分类的LDA模型降维并训练
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_race = LDA(n_components=6)
lda_race.fit(xtrain,ytrain[:,3])
#将训练好的LDA模型对测试数据进行特征提取
xtest_race = lda_race.transform(xtest)
print('用训练后的种族分类LDA模型降维后的图像测试数据：',xtest_race.shape)

#创建表情分类的LDA模型降维并训练
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_face = LDA(n_components=2)
lda_face.fit(xtrain,ytrain[:,4])
#将训练好的LDA模型对测试数据进行特征提取
xtest_face = lda_face.transform(xtest)
print('用训练后的表情分类LDA模型降维后的图像测试数据：',xtest_face.shape)