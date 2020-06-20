# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 21:59:11 2020

@author: Lenovo
"""

# Logistic Regression

import numpy as np

class LogisticRegression():
    def __init__(self):
        self.predictlabel = []
        
    def sigmoid(self, x):
        return 1.0 / (1+np.exp(-x))
    
    def training(self, trainset, trainlabel, alpha, iteration):
        # Initialize
        self.dataSet = np.array(trainset)
        self.label = np.array(trainlabel)
        self.weight = np.zeros(self.dataSet.shape[1])
        
        for i in range(iteration):
            for j in range(self.dataSet.shape[0]):
                h = self.sigmoid(np.dot(self.dataSet[j], self.weight))
                error = self.label[j] - h
                self.weight += alpha * error * self.dataSet[j] * (h-0.5)**2
                
    def test(self, testset):
        for item in testset:
            h = self.sigmoid(np.dot(item, self.weight))
            label = 0
            if h >= 0.5:
                label = 1
            self.predictlabel.append(label)
        return self.predictlabel