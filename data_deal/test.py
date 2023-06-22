# import pandas as pd  # 导入pandas库
#
# fm = pd.read_excel(r"D:\Desktop\faceDS.xlsx", sheet_name="Sheet1")  # 用该方法读取表格和表单里的单元格的数据
# print(fm)
# print(fm[1])
def use_xls():
    import xlrd#导入xlrd库
    file='D:/Desktop/faceDS.xls'#文件路径
    wb=xlrd.open_workbook(filename=file)#用方法打开该文件路径下的文件
    #wb = open('D:/Desktop/faceDS.xlsx')
    ws=wb.sheet_by_name("Sheet1")#打开该表格里的表单
    dataset=[]
    for r in range(ws.nrows):#遍历行
        col=[]
        for l in range(ws.ncols):#遍历列
            col.append(ws.cell(r, l).value)#将单元格中的值加入到列表中(r,l)相当于坐标系，cell（）为单元格，value为单元格的值
        dataset.append(col)
    from pprint import pprint#pprint的输出形式为一行输出一个结果，下一个结果换行输出。实质上pprint输出的结果更为完整
    #pprint(dataset)
    print(dataset[1][1])
    return dataset

use_xls()