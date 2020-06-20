# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:03:31 2020

@author: Lenovo
"""

import numpy as np

class SVM():
    def __init__(self):
        self.C = 100    # 软间隔
        self.dimension = 0
        self.trainData = np.array([[]])
        self.label = np.array([])
        self.kernelMatrix = np.array([[]])
        self.b = 0
        self.epsilon = 0.00001
        self.predictlabel = []
        
    def RBF(self, item, vector, sigma):     # 高斯核，即径向基函数
        return np.exp( -np.square(np.linalg.norm(np.abs(item - vector)))
                                                      /(2 * sigma**2) )
        
    def error(self, i):
        return np.sum([self.alpha[j]*self.label[j]*self.kernelMatrix[i, j] 
                           for j in range(self.dimension)]) + self.b - self.label[i]
                                
    def KKTcheck(self, i, epsilon, C):
        temp = self.label[i] * ( np.sum([self.alpha[j]*self.label[j]*
                            self.kernelMatrix[i, j] for j in range(self.dimension)])
                            + self.b )
        if ((-epsilon < self.alpha[i] < epsilon) and (temp >= 1 - epsilon)):
            return 1
        elif ((-epsilon < self.alpha[i] < C + epsilon) and (1 - epsilon <= temp <= 1 + epsilon)):
            return 1
        elif ((C - epsilon < self.alpha[i] < C + epsilon) and (temp <= 1 + epsilon)):
            return 1
        else:
            return 0
    
    def training(self, trainset, trainlabel, sigma, C):
        # Initialize
        self.C = C
        self.trainData = np.array(trainset)
        self.label = np.array(trainlabel)
        self.dimension = self.trainData.shape[0]
        self.alpha = np.zeros(self.dimension)
        self.kernelMatrix = np.zeros([self.dimension, self.dimension], 
                                         dtype=float)
        self.sigma = sigma
        self.E = []
        # 计算核函数表
        for x in range(self.dimension):
            for y in range(self.dimension):
                self.kernelMatrix[x, y] = self.RBF(self.trainData[x], 
                                                 self.trainData[y], sigma)
                self.kernelMatrix[y, x] = self.RBF(self.trainData[x], 
                                                 self.trainData[y], sigma)
        
        for i in range(self.dimension):
            self.E.append(-self.label[i])
        
        # 开始迭代
        flag = 0    # KKT条件是否满足
        loop = 0
        while flag == 0:
            flag = 1
            loop += 1
            print(loop)
            for i in range(self.dimension):
                if self.KKTcheck(i, self.epsilon, self.C) == 0:
                    E1 = self.E[i]
                    maxdiff = -1
                    for s in range(self.dimension):
                        tempdiff = np.fabs(self.E[s] - E1)
                        if tempdiff > maxdiff:
                            maxdiff = tempdiff
                            E2 = self.E[s]
                            j = s
                    if maxdiff == -1:   # 不存在使得目标函数下降的点
                        continue
                    
                    # 求上下界
                    H = 0
                    L = self.C
                    # L
                    if self.label[i] == self.label[j]:
                        L = self.alpha[i] + self.alpha[j] - self.C
                    else:
                        L = self.alpha[j] - self.alpha[i]
                    if L <= 0:
                        L = 0
                    # H
                    if self.label[i] == self.label[j]:
                        H = self.alpha[i] + self.alpha[j]
                    else:
                        H = self.alpha[j] - self.alpha[i] + self.C
                    if L <= 0:
                        H >= self.C
            
                    # 求newalpha2
                    newalpha2 = self.alpha[j] + \
                    self.label[j]*(E1-E2)/(self.kernelMatrix[i, i]+ \
                              self.kernelMatrix[j, j]-2*self.kernelMatrix[i, j])
                    if newalpha2 < L:
                        newalpha2 = L
                    if newalpha2 > H:
                        newalpha2 = H
                    # 求newalpha1
                    newalpha1 = self.alpha[i] + self.label[i] * self.label[j] \
                        * (self.alpha[j] - newalpha2)
                        
                    # 求newb
                    newb1 = -E1 - self.label[i] * self.kernelMatrix[i, i] * (
                        newalpha1 - self.alpha[i]) - self.label[j] * \
                        self.kernelMatrix[j, i] * (newalpha2 - self.alpha[j]) \
                        + self.b
                    newb2 = -E2 - self.label[i] * self.kernelMatrix[i, j] * (
                        newalpha1 - self.alpha[i]) - self.label[j] * \
                        self.kernelMatrix[j, j] * (newalpha2 - self.alpha[j]) \
                        + self.b
        
                    # 判断迭代条件
                    if (np.fabs(self.alpha[i] - newalpha1) < self.epsilon ** 2) \
                        and (np.fabs(self.alpha[j] - newalpha2) < self.epsilon ** 2):
                        continue
                    else:
                        flag = 0
                    
                    # 更新alpha、b、E
                    self.alpha[i] = newalpha1
                    self.alpha[j] = newalpha2

                    if 0 < newalpha1 < self.C:
                        self.b = newb1
                    elif 0 < newalpha2 < self.C:
                        self.b = newb2
                    else:
                        self.b = (newb1 + newb2) / 2
                        
                    self.E[i] = self.error(i)
                    self.E[j] = self.error(j)
        
    def test(self, testset):
        for item in testset:
            differences = []
            for i in range(self.dimension):
                if self.alpha[i] > 0:
                    difference = self.alpha[i] * self.label[i] * self.RBF(item, self.trainData[i], self.sigma)
                    differences.append(difference)
            differences = np.array(differences)
            dist = np.sum(differences) + self.b
            label = np.sign(dist)
            self.predictlabel.append(label)
        return self.predictlabel
        
        
        