# -*- coding: utf-8 -*-
import sklearn
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy


def getData_1():

    iris = datasets.load_iris()
    X = iris.data   #样本特征矩阵，150*4矩阵，每行一个样本，每个样本维度是4
    y = iris.target #样本类别矩阵，150维行向量，每个元素代表一个样本的类别


    df1=pd.DataFrame(X, columns =['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
    df1['target']=y

    return df1

df=getData_1()


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:3],df['target'], test_size=0.3, random_state=42)
print(X_train, X_test, y_train, y_test)

model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001,max_iter=10000)  # 神经网络
"""参数
---
    n_neighbors： 使用邻居的数目
    n_jobs：并行任务数
"""
model.fit(X_train,y_train)
predict=model.predict(X_test)
print (predict)
print(y_test.values)

print('神经网络分类:{:.3f}'.format(model.score(X_test, y_test)))
