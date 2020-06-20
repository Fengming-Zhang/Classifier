# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:40:48 2020

@author: Lenovo
"""

import numpy as np
from sklearn import preprocessing
import csv

from KNN import KNN
from SVM import SVM
from other import LogisticRegression

class DataSet():
    def __init__(self):
        self.column = []
        self.data = [[] for i in range(32)]
        self.dataSet = np.array([[]])
        self.label = []
        self.trainset = np.array([[]])
        self.testset = np.array([[]])
        self.trainlabel = np.array([])
        self.testlabel = np.array([])
        
    
    def preprocessing(self, filename, scale, usingG, isSVM):
        labelEncoder = preprocessing.LabelEncoder()
        with open(filename, 'r') as sourcefile:
            lines = csv.reader(sourcefile, delimiter = ';')
            cnt = 0
            # 使用G1、G2
            if usingG:
                for line in lines:
                    if cnt == 0:
                        self.column = line
                    else:
                        for i in range(32):
                            if i in [2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 28
                                     , 29, 30, 31]:
                                self.data[i].append(int(line[i]))
                            else:
                                self.data[i].append(line[i])
                        if int(line[32]) >= 10:
                            self.label.append(1)
                        else:
                            if isSVM:
                                self.label.append(-1)
                            else:
                                self.label.append(0)
                    cnt += 1
                for i in range(32):
                    if i in [2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 28
                                     , 29, 30, 31]:
                        continue
                    self.data[i] = labelEncoder.fit_transform(self.data[i])
            else:   
                self.data = self.data = [[] for i in range(30)]
                for line in lines:
                    if cnt == 0:
                        self.column = line
                    else:
                        for i in range(30):
                            if i in [2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 28
                                     , 29]:
                                self.data[i].append(int(line[i]))
                            else:
                                self.data[i].append(line[i])
                        if int(line[32]) >= 10:
                            self.label.append(1)
                        else:
                            if isSVM:
                                self.label.append(-1)
                            else:
                                self.label.append(0)
                    cnt += 1
                for i in range(30):
                    if i in [2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 28
                                     , 29]:
                        continue
                    self.data[i] = labelEncoder.fit_transform(self.data[i])
        # 需要shuffle随机化

        tempDataSet = self.data
        tempDataSet.append(self.label)
        tempDataSet = np.array(tempDataSet).T
        np.random.shuffle(tempDataSet)
        # print(tempDataSet)
        self.dataSet, self.label = np.split(tempDataSet, [-1], 1)
        self.dataSet = np.array(self.dataSet)
        self.label = self.label.flatten()
        self.label = self.label.tolist()
        
        # split
        splitPos = int(cnt * scale)
        for line in range(cnt):
            self.trainset = self.dataSet[ :splitPos]
            self.testset = self.dataSet[splitPos+1: ]
            self.trainlabel = self.label[ :splitPos]
            self.testlabel = self.label[splitPos+1: ]
        # print(self.data)
        # print(self.label)
        # print(self.dataSet)
        # print(cnt)
        print(self.trainset)
        print(self.testset)
        print(self.trainlabel)
        print(self.testlabel)


def evaluation(testlabel, predictlabel):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(testlabel)):
        if testlabel[i] == 1 and predictlabel[i] == 1:
            tp += 1
        if testlabel[i] != 1 and predictlabel[i] == 1:
            fp += 1
        if testlabel[i] == 1 and predictlabel[i] != 1:
            fn += 1
        if testlabel[i] != 1 and predictlabel[i] != 1:
            tn += 1
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (P + R)
    
    print("准确率 P = " + str(P))
    print("召回率 R = " + str(R))
    print("F1 score = " + str(F1))
            
            
if __name__ == "__main__":
    matfile = "../data/student-mat.csv"
    porfile = "../data/student-por.csv"
    dataset = DataSet()
    
    method = input("Method: ")
    # method = 0
    
    if method == 'knn':
        print("---------running KNN------------")
        knn = KNN()
        dataset.preprocessing(porfile, 0.7, usingG=1, isSVM=0)
        predictlabel = knn.training(trainset=dataset.trainset, 
                                    testset=dataset.testset, 
                                    trainlabel=dataset.trainlabel, K=9)
        print(predictlabel)
        evaluation(dataset.testlabel, predictlabel)
    
    elif method == 'svm':
        print("---------running SVM------------")
        svm = SVM()
        dataset.preprocessing(matfile, 0.7, usingG=1, isSVM=1)
        svm.training(trainset=dataset.trainset, trainlabel=dataset.trainlabel,
                     sigma=5, C=100)
        predictlabel = svm.test(testset=dataset.testset)
        print(predictlabel)
        evaluation(dataset.testlabel, predictlabel)
        
    elif method == 'other':
        print("---------running ------------")
        lr = LogisticRegression()
        dataset.preprocessing(porfile, 0.7, usingG=0, isSVM=0)
        lr.training(trainset=dataset.trainset, trainlabel=dataset.trainlabel, 
                    alpha=0.005, iteration=200)
        predictlabel = lr.test(testset=dataset.testset)
        print(predictlabel)
        evaluation(dataset.testlabel, predictlabel)
    else:
        print("Error: Invalid method!")
        
        
        
        
        
        
        