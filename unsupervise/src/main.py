# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:16:32 2020

@author: Lenovo
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

from PCA import PCA
from Kmeans import Kmeans

class DataSet():
    def __init__(self):
        self.dataSet = np.array([[]])
        self.data = [[] for i in range(13)]
        self.label = []
        
    def preprocessing(self, filename, scale=0.7):
        cnt = 0
        with open(filename, 'r') as sourcefile:
            lines = csv.reader(sourcefile, delimiter = ',')
            for line in lines:
                self.label.append(int(line[0]))
                for i in range(13):
                    self.data[i].append(float(line[i+1]))
                cnt += 1
        # 标准化
        for i in range(13):
            self.data[i] = np.array(self.data[i]) - np.min(np.array(self.data[i]))
            self.data[i] = self.data[i].tolist()
        # shuffle
        tempDataSet = self.data
        tempDataSet.append(self.label)
        tempDataSet = np.array(tempDataSet).T
        np.random.shuffle(tempDataSet)
        # print(tempDataSet)
        self.dataSet, self.label = np.split(tempDataSet, [-1], 1)
        self.dataSet = np.array(self.dataSet)
        self.label = self.label.flatten()
        self.label = self.label.tolist()
#        print(self.label)
        
        
def evaluate(label, predictlabel, dataset):
    cnt = len(label)
    # RI
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(cnt):
        for j in range(cnt):
            if i == j:
                continue
            elif label[i] == label[j] and predictlabel[i] == predictlabel[j]:
                a += 1
            elif label[i] == label[j] and predictlabel[i] != predictlabel[j]:
                b += 1
            elif label[i] != label[j] and predictlabel[i] == predictlabel[j]:
                c += 1
            elif label[i] != label[j] and predictlabel[i] != predictlabel[j]:
                d += 1
    RI = (a+d) / (a+b+c+d)
    print("兰德系数: RI = " + str(RI))
    
    # S
    S = []
    lista = []
    listb = []
    for i in range(cnt):
        distSamePoints = []
        distNsamePoints = []
        for j in range(cnt):
            if i != j and predictlabel[i] == predictlabel[j]:
                dist = np.linalg.norm(dataset[i] - dataset[j])
                distSamePoints.append(dist)
            elif i != j:
                dist = np.linalg.norm(dataset[i] - dataset[j])
                distNsamePoints.append(dist)
        lista.append(np.mean(np.array(distSamePoints)))
        listb.append(np.mean(np.array(distNsamePoints)))
    for i in range(cnt):
        Si = (listb[i]-lista[i]) / max([lista[i], listb[i]])
        S.append(Si)
    SilhouetteCoefficient = np.mean(np.array(S))
    print("轮廓系数: S = " + str(SilhouetteCoefficient))


def visualize(dataset, label, K=3):
    redX, redY = [], []
    blueX, blueY = [], []
    greenX, greenY = [], []
    for i in range(len(dataset)):
        if label[i] == 0:
            redX.append(dataset[i][0])
            redY.append(dataset[i][1])
        elif label[i] == 1:
            blueX.append(dataset[i][0])
            blueY.append(dataset[i][1])
        elif label[i] == 2:
            greenX.append(dataset[i][0])
            greenY.append(dataset[i][1])
    plt.scatter(redX, redY, c='r', marker='x')
    plt.scatter(blueX, blueY, c='b', marker='D')
    plt.scatter(greenX, greenY, c='g', marker='.')
    plt.show()
    

if __name__ == "__main__":
    winefile = "../input/wine.csv"
    print("---------Preprocessing------------")
    dataset = DataSet()
    dataset.preprocessing(winefile)
    
    print("---------PCA------------")
    pca = PCA()
    dimmedset = pca.fit(data=dataset.dataSet)
    # print(dimmedset)
    
    print("---------K-means------------")
    kmeans = Kmeans()
    print("---------不采用降维数据------------")
    predictlabel = kmeans.cluster(dataset.dataSet)
#    print(predictlabel)
    evaluate(dataset.label, predictlabel, dataset.dataSet)
    print("---------采用降维数据------------")
    print("维数: dim = " + str(pca.dim))
    predictlabel = kmeans.cluster(dimmedset)
#    print(predictlabel)
    evaluate(dataset.label, predictlabel, dimmedset)
    if pca.dim == 2:
        visualize(dimmedset, predictlabel)
    
    
    
    