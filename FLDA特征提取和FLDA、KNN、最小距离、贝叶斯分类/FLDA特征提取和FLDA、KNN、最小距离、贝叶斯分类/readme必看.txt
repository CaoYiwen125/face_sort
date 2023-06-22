请将数据集和py、notebook文件放在同一文件夹中，不需要更改代码中的地址。

faceDR获取到的字符串列表中，前面会多一个多余的字符串，需要先用del处理一下。记得np.array之前先打印出来检查。
faceDR中有拼写错误，从faceDR获得ytrain后，请记得用spellError.py中的代码处理。

FLDA_extract.py：FLDA特征提取
FLDA_classify.py：FLDA模型直接用于分类及其交叉验证
FLDA_Bayes.py：FLDA特征提取+Bayes分类+交叉验证
FLDA_KNN.py：FLDA特征提取+KNN分类+交叉验证
FLDA_MinimumDistanceClasifier.py：FLDA特征提取+最小距离分类+评分

可以先看jupyter notebook的文件比较清楚。
flda特征提取即分类.ipynb：包含了FLDA_extract.py、FLDA_classify.py、FLDA_Bayes.py、FLDA_KNN.py的代码和文字说明
flda_最小距离分类器.ipynb：包含了FLDA_MinimumDistanceClasifier.py的代码和文字说明