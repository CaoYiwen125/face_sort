import numpy as np
import copy
h1 = open("D:/Desktop/faceDR",'r',encoding = 'utf-8')#⭐换时该路径
# h1 = open("D:/Desktop/faceDS",'r',encoding = 'utf-8')#⭐换时该路径
start_num = 1223                 #⭐换时要更改
end_num = 3222                   #⭐换时要更改
guiyi = 0                        #⭐是否要归一化？要归一化改1，不要改0(不过y，也就是对应结果要归一化的情况没看到过，故默认为0，一般是不会改的)
##########################################⭐只要改上面四个就行了
'''
特别说明，这里是数值化部分，有问题请问译文。
这里中，
性别sex——————female：1；male：2
年龄age——————child:1,teen:2,adult:3,senior:4
种族race——————white:1,asian:2,hispanic:3,black:4,other:5
表情face——————serious:1,smiling:2,funny:3
输出的结果矩阵中，
[:,0]-------照片编号
[:,1]-------sex
[:,2]-------age
[:,3]-------race
[:,4]-------face
'''
datanum = end_num-start_num+1
print(h1)
h2 = h1.read()
# print(h2)
# Python3 code to demonstrate working of
# Delimited String List to String Matrix
# Using loop + split()
# initializing list
# test_list = ['gfg:is:best', 'for:all', 'geeks:and:CS']
# printing original list
# print("The original list is : " + str(test_list))
# Delimited String List to String Matrix
# Using loop + split()
h3 = h2
# print(h3.split(' '))
h4 = h3.split(' ')
# print(h4[14])
# res = []
# for sub in h2:
#     res.append(sub.split('_'))
# printing result
# print("The list after conversion : " + str(res))
# sex=[];age=[];race=[];face=[];prop=[];
#DATA = []
# DATA = torch.zeros(2000, 6)
#性别sex数字化
def sex_num(sex):
    if sex=='female':
        getnum1 = 1.0
    if sex=='male':
        getnum1 = 2.0
    return getnum1
#年龄age数字化
def age_num(age):
    if age=='child':
        getnum2 = 1.0
    if age=='teen':
        getnum2 = 2.0
    if age=='adult':
        getnum2 = 3.0
    if age=='senior':
        getnum2 = 4.0
    return getnum2

#种族race数字化
def race_num(race):
    if race=='white':
        getnum3 = 1.0
    if race=='asian':
        getnum3 = 2.0
    if race =='hispanic':
        getnum3 = 3.0
    if race=='black':
        getnum3 = 4.0
    if race=='other':
        getnum3 = 5.0
    return getnum3

#表情face数字化
def face_num(face):
    if face=='smiling':
        getnum4 = 2.0
    if face=='serious':
        getnum4 = 1.0
    if face=='funny':
        getnum4 = 3.0
    return getnum4


DATA = np.full((datanum,5),0)
# print(len(h4))
num = 1
i = 0
# h = 0
# cout = 0
while num <= len(h4):
# while cout < 3222:
    cout = start_num+i
    if cout > end_num:
        break
    cout=str(cout)
    # print(cout)
    #print(len(h4))
    if h4[num] == cout:
        DATA[i][0] = h4[num]
        # int(cout)
        # print('get',num)
        num+=1
        if h4[num]=='(_sex':
            num+=2
            # print('sex',num)
            sexn = sex_num(h4[num].replace(')',''))
            DATA[i][1] = sexn
            num+=3
            # print('age',num)
            agen = age_num(h4[num].replace(')', ''))
            DATA[i][2] = agen
            num+=2
            # print('race',num)
            racen = race_num(h4[num].replace(')', ''))
            DATA[i][3] = racen
            num+=2
            # print('face',num)
            facen = face_num(h4[num].replace(')', ''))
            DATA[i][4] = facen
            # print('REAL_DATA',DATA[h,:])
            num+=2
            # DATA2=DATA
            # print('DATA2',DATA2)
            # print('prop',num)
            # DATA[i][5] = h4[num].replace(')','')
        # else:
        #    continue
            num += 1
            i += 1
            #
            # h+=1
            #
        else:
            num+=1
            i += 1
            # print(DATA[1999, :])
            #
            # h=h
    else:
        num+=1
        # print(DATA[1999, :])
    # DATA2=DATA
    # print(DATA[1999,:])
print('FINISH!⭐数据预处理（结果数字化）已完成！')
# print('DATA',DATA)
# print('DATA2',DATA2)
# print(DATA[5,:])
# print(h,i)
DATAR = copy.deepcopy(DATA)
# print(type(DATAR))
allong = len(DATAR[:,0])
# print(allong)
i = 0
h = 0
delnum = 0
while h < allong:
    if DATAR[i,1]==0:
        DATAR = np.delete(DATAR,i,axis=0)
        print('删了',h+start_num)
        i-=1
        delnum+=1

    i+=1
    h+=1
    # print(i)
print('删了错误行行数：',delnum)
if guiyi==1:############3guiyi
    maxsex = max(DATAR[:,1])
    minsex = min(DATAR[:,1])
    DATAR[:,1]=(DATAR[:,1]-minsex)/(maxsex-minsex)

    maxage = max(DATAR[:,2])
    minage = min(DATAR[:,2])
    DATAR[:,2]=(DATAR[:,2]-minage)/(maxage-minage)

    maxrace = max(DATAR[:,3])
    minrace = min(DATAR[:,3])
    DATAR[:,3]=(DATAR[:,3]-minrace)/(maxrace-minrace)


    maxface = max(DATAR[:,4])
    minface = min(DATAR[:,4])
    DATAR[:,4]=(DATAR[:,4]-minface)/(maxface-minface)
    # print(maxface)
print('DATAR:')
print(DATAR)
h1.close()