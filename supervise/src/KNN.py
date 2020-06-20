# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:09:02 2020

@author: Lenovo
"""

import numpy as np


class KNN():
    def __init__(self):
        self.predictlabel = []

    def training(self, trainset, testset, trainlabel, K):
        for item in testset:
            distances = []
            for vector in trainset:
                distance = np.linalg.norm(np.abs(item-vector)) # 二范数，即模长
                distances.append(distance)
            distances = np.array(distances)
            neighbors = np.argpartition(distances, K) # 返回索引
            K_neighbors = neighbors[:K]
            
            # 计数器，记录K近邻的标签pass和fail的数目
            prePass = 0
            preFail = 0
            for neighbor in K_neighbors:
                if trainlabel[neighbor] == 1:
                    prePass += 1
                else:
                    preFail += 1
            label = 0
            if prePass >= preFail:
                label = 1
            self.predictlabel.append(label)
        return self.predictlabel