import numpy as np

# 读取原始图像数据
with open('D:/Desktop/newraw/'+str(1223)+'.png', 'r') as fid:
    data = fid.read()
    # print(type(fid))
    # print(type(data))
    # print(data)
I = np.frombuffer(data,dtype=np.uint8)
# print('I1:',I)
I = I.reshape((128,128))
# print('I1:',I)
# 将图像数据转换为浮点型，并进行归一化处理
I = I.astype(np.float32) / 255.0

# 将图像数据重塑为128x128的矩阵
# I = np.reshape(I, (128, 128))

# 计算图像数据的平均值和标准差
mean = np.mean(I)
std = np.std(I)

# 对图像数据进行标准化处理
I = (I - mean) / std
print(I)
# 提取图像数据的特征向量，并截取前100个元素
descriptor = np.ravel(I)[:100]

# 打印特征向量
print(' '.join([str(x) for x in descriptor]))