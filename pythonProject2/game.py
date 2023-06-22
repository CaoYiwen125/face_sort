import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
# 读取多维数组 faceR
faceR = np.loadtxt(r'D:\Desktop\faceR') #faceR

# 生成相应数量的标签
labels = np.arange(faceR.shape[0])
print(labels)#结果0~1999`
# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(faceR)#标准化

# 创建PCA对象并进行降维
pca = PCA(n_components=2)#这里需调试
X_pca = pca.fit_transform(X_scaled)

# 打印降维后的维度
print("PCA result shape:", X_pca.shape)

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=100,n_init='auto')  # 设置聚类簇数为3，根据实际情况调整
kmeans.fit(X_pca)

# 获取聚类结果的标签
cluster_labels = kmeans.labels_
print(cluster_labels)#[39 80 97 ... 1 84 19]不过结果是不定的,但长度也为2000
print(len(cluster_labels))#2000
print(max(cluster_labels))#99
# 将聚类结果的标签与独立标签合并作为最终的标签
labels = np.column_stack((labels, cluster_labels))
#这个column_stack是两个矩阵按列合并成
#labels = [0 39][1 80]...[1999 19]
#保存标签结果到文件（例如 'labels.txt'）
np.savetxt('labels.txt', labels, fmt='%d')
print(labels)
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, labels, test_size=0.2, random_state=2)

# 实例化SVM
svm = SVC(kernel='linear', C=10)

# 训练线性SVM
svm.fit(X_train, y_train[:, 1])  # 使用合并后的标签作为目标变量

# 预测
y_pred = svm.predict(X_test)

print(classification_report(y_test[:, 1], y_pred))
#“精确率”，“召回率”，“F1值”，样本数“”
# 可视化聚类结果
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clustering Result')
plt.show()
#print(dir(f1))