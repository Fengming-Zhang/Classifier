# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:36:44 2020

@author: Lenovo
"""

import numpy as np

class PCA():
    def __init__(self):
        self.trainset = np.array([[]])
        
    def fit(self, data, threshold = 0.999):
        self.trainset = np.array(data)    # m × n
        normMatrix = self.trainset - self.trainset.mean(axis = 0)
        covMatrix = np.cov(normMatrix.T)    # 转置为 n × m
        # covMatrix为 n × n, 求特征值
        eigenvalue, eigenvector = np.linalg.eig(covMatrix)
#        print("eigenVector")
#        print(eigenvector)
        templist = sorted(zip(eigenvalue, eigenvector), key=lambda x:(x[0]), reverse=True)
#        print("templist")
#        print(templist)
        eigenvalue, eigenvectorTuple = zip(*templist)
        
        orderedEigenvector = []
        for i in range(len(eigenvalue)):
            orderedEigenvector.append(eigenvectorTuple[i].tolist())
#        print("eigenValue")
#        print(eigenvalue)
#        print("orderedEigenvector")
#        print(orderedEigenvector)
        # n个特征值， n个n维特征向量
        # 其中选前k个
        sumEigen = sum(eigenvalue)
        sumtemp = 0
        chosenVector = []
        for i in range(len(eigenvalue)):
            sumtemp += eigenvalue[i]
            chosenVector.append(orderedEigenvector[i])
            if sumtemp / sumEigen > threshold:
                break
        
#        print("chosenVector")
#        print(chosenVector)
        transMatrix = np.array(chosenVector).T
#        print("trainset")
#        print(self.trainset)
#        print("transMatrix")
#        print(transMatrix)
        self.dimmedset = np.dot(self.trainset, transMatrix)
        self.dim = self.dimmedset.shape[1]
        return self.dimmedset
        
        
        