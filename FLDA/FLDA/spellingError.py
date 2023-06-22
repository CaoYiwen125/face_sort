import numpy as np
#faceDR中错误修改(faceDS无拼写错误)
##faceDR##
###faceDR###
f = open("faceDR","r",encoding="utf8")
y_1 = f.read()
f.close()
y_1 = y_1.split() #默认删除所有空字符，包括空格和换行符等
#print(y_1)
del y_1[0]
#print(y_1)
y_1 = np.array(y_1)
y_1.resize((1996,5))
ytrain = y_1
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
print('修改拼写错误后的训练标签数据ytrain：\n', ytrain)
