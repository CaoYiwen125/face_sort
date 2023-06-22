一。请将数据集和py、notebook文件放在同一文件夹中，不需要更改代码中的地址。

二。faceDR获取到的字符串列表中，前面会多一个多余的字符串，需要先用del处理一下。记得np.array之前先打印出来检查。
从faceDR获得ytrain后，有拼写错误，spellError.py记录了修改代码，拼写错误修改.ipynb记录了检查和修改过程。

目前已将拼写改错代码放进特征提取部分中，改动了准确的降维维数，更新降维数据。下面提到的“特征提取”都已包含了拼写改错。

三。
FLDA_extract.py：FLDA特征提取
FLDA_classify.py：FLDA模型直接用于分类及其交叉验证
FLDA_Bayes.py：FLDA特征提取+Bayes分类+交叉验证
FLDA_KNN.py：FLDA特征提取+KNN分类+交叉验证
FLDA_MinimumDistanceClasifier.py：FLDA特征提取+最小距离分类+评分
FLDA_SVM.py：FLDA特征提取+SVM分类+交叉验证
FLDA_BP.py：FLDA特征提取+BP网络分类+交叉验证
FLDA_DT.py：FLDA特征提取+决策树分类+交叉验证

可以先看jupyter notebook的文件比较清楚。
flda特征提取_FLDA分类_Bayes_KNN.ipynb：包含了FLDA_extract.py、FLDA_classify.py、FLDA_Bayes.py、FLDA_KNN.py的代码和文字说明
flda_最小距离分类器.ipynb：包含了FLDA_MinimumDistanceClasifier.py的代码和文字说明
flda特征提取_SVM_BP_DT.ipynb：包含了FLDA_SVM.py、FLDA_BP.py、FLDA_DT.py的代码和文字说明

四。py文件中可能会有漏掉模块导入、变量定义等问题，如若有，补一下就好。