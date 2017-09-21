# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:13:20 2017

@author: hyyh6
"""

import os
import fnmatch
import FileDefines as fd
import random

def IterFindFiles(path, fnexp):
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, fnexp):
            yield os.path.join(root, filename)


trainFile = open(fd.TRAIN_CSV_FILE, "w")
testFile = open(fd.TEST_CSV_FILE, "w")
trainData = []
testData = []

def GenerateWithLimit(train_limit, test_limit):
    for i in range(10) :
        fileFilter = "*" + str(i) + ".png"
        
        j = 0
        for filename in IterFindFiles(r"./Samples/train", fileFilter) :
            print filename
            pos = filename.find("train")
            if pos >= 0 and j < 4000:
                j = j+1
                strLen = len(filename)
                numChar = filename[strLen-5]
                labelStr = "," + numChar + "\n"
                strFile = filename + labelStr
                trainData.append(strFile)     
        
        j = 0
        for filename in IterFindFiles(r"./Samples/test", fileFilter) :
            pos = filename.find("test")
            if pos >= 0 and j < 1400 :
                j = j+1
                strLen = len(filename)
                numChar = filename[strLen-5]
                labelStr = "," + numChar + "\n"
                strFile = filename + labelStr
                testData.append(strFile)
                
    random.shuffle(trainData)
    random.shuffle(testData)
    
    for strFile in trainData:
        trainFile.write(strFile)
    
    for strFile in testData:
        testFile.write(strFile)

    trainFile.close()
    testFile.close()          

def Generate():
    for filename in IterFindFiles(r"./Samples", "*.png") :
        print filename
        pos = filename.find("train")
        if pos >= 0 :
            strLen = len(filename)
            numChar = filename[strLen-5]
            labelStr = "," + numChar + "\n"
            strFile = filename + labelStr
            trainData.append(strFile)
    
        pos = filename.find("test")
        if pos >= 0 :
            strLen = len(filename)
            numChar = filename[strLen-5]
            labelStr = "," + numChar + "\n"
            strFile = filename + labelStr
            testData.append(strFile)
    
    random.shuffle(trainData)
    random.shuffle(testData)
    
    for strFile in trainData:
        trainFile.write(strFile)
    
    for strFile in testData:
        testFile.write(strFile)
    
    trainFile.close()
    testFile.close()
    
#GenerateWithLimit(4000, 1400)
Generate()