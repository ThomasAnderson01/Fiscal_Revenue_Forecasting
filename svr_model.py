# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:15:03 2019

@author: chern.lei
"""


import pandas as pd
from sklearn.svm import LinearSVR

inputFile = 'new_reg_data_GM11.xls'               # 灰色预测后保存的路径
data = pd.read_excel(inputFile, index_col=0)      # 读取数据
feature = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']
data_train = data.loc[range(1994, 2014)].copy()   # 取2014年前的数据建模

data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std   # 数据标准化
x_train = data_train[feature].as_matrix()        # 特征数据
y_train = data_train['y'].as_matrix()            # 标签数据

svr = LinearSVR()                                # 调用LinearSVR()函数
svr.fit(x_train, y_train)                        # 模型训练

x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()  # 预测，并还原结果。
data[u'y_pred'] = svr.predict(x) * data_std['y'] + data_mean['y']

outputFile = 'new_reg_data_GM11_revenue.xls'     # SVR预测后保存的结果
data.to_excel(outputFile)

print('真实值与预测值分别为：', data[['y', 'y_pred']])
print('预测图为：', data[['y', 'y_pred']].plot(subplots=True, style=['b-o', 'r-*']))
