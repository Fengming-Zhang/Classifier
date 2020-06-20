# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 21:42:48 2020

@author: Lenovo
"""

import numpy as np
import random

class Kmeans():
    def __init__(self):
        self.trainset = np.array([[]])
        self.predictlabel = []
        
    def cluster(self, data, K=3, epsilon=0.5):
        self.trainset = np.array(data)
        self.count = self.trainset.shape[0]
        self.dim = self.trainset.shape[1]
        self.K = K
        # 初始K个中心点
        centers = []
        for i in range(K):
            index = random.randint(0, self.count-1)
            centers.append(self.trainset[index])
            
        # 开始迭代
        predictlabel = []
        offset = 10
        loop = 0
        while offset > 0.5:
            loop += 1
            predictlabel = []
            print(loop)
            for i in range(self.count):
                dist = 100000000
                label = 0
                for curlabel in range(K):
                    tempdist = np.linalg.norm(self.trainset[i] - centers[curlabel])
                    if dist > tempdist:
                        dist = tempdist
                        label = curlabel
                predictlabel.append(label)
            
            # 更新center
            centersNew = []
            for curlabel in range(K):
                nearpoints = []
                for i in range(self.count):
                    if predictlabel[i] == curlabel:
                        nearpoints.append(self.trainset[i])
                centerNew = np.mean(np.array(nearpoints), axis=0)
                centersNew.append(centerNew)
                
            # 计算offset
            offset = 0
            for curlabel in range(K):
                offset += np.linalg.norm(centersNew[curlabel] - centers[curlabel])
            centers = centersNew.copy()
        
        self.predictlabel = predictlabel
        self.centers = centers
        
        return self.predictlabel
            
            
            
                        