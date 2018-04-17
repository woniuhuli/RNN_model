# -*- coding: utf-8 -*-
'''
@输入参数：trainData文件路径
@输出：二维数组[time,flavors]，第一维为日期，第二维为当天flavor1到flavor15的销量
'''
import datetime
import os

def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        print('file not exist: ' + file_path)
        return None

def ReadTrainData(dataPath):
    # @dataPath:    trainData文件路径
    # @返回：  二维数组[time,flavors]，第一维为日期，第二维为当天flavor1到flavor15的销量
    lines = read_lines(dataPath)
    time_start = datetime.datetime.strptime(lines[0].split("\t")[2].split(" ")[0],'%Y-%m-%d')
    time_end = datetime.datetime.strptime(lines[-1].split("\t")[2].split(" ")[0],'%Y-%m-%d')
    days = (time_end - time_start).days + 1 #训练集天数
    TrainData = [[0 for i in range(15)] for j in range(days)] #二维数组初始化
    for line in lines:
        day = (datetime.datetime.strptime(line.split("\t")[2].split(" ")[0],'%Y-%m-%d') - time_start).days
        flavor = line.split("\t")[1]
        if flavor[-2] == '1' and int(flavor[-1])<=5:
            flavor = int(flavor[-2:])-1
            TrainData[day][flavor] += 1
        else:
            if flavor[-2] != '2':
                flavor = int(flavor[-1])-1
                TrainData[day][flavor] += 1
    return TrainData

def ReadInputData(dataPath):
    # @dataPath:  inputData文件路径
    # @flavorToPredict: 待预测规格，数组
    # @flavorDict:      待预测规格，字典
    # @timeToPredict:   待预测天数，整数
    # @PhysicalServer:  物理服务器容量，字典
    # @target：         优化目标，字符串
    lines = read_lines(dataPath)
    physicalInfo = lines[0].strip('\n').split(" ")
    physicalServer = {'CPU': int(physicalInfo[0]), 'MEM': int(physicalInfo[1]), 'HAD': int(physicalInfo[2])}
    flavors = int(lines[2].strip("\n"))
    flavorToPredict = []
    flavorDict = {}
    for i in range(flavors):
        flavorInfo = lines[i+3].strip("\n").split(" ")
        flavorDict[flavorInfo[0]] = flavorInfo[1:]
        flavor = flavorInfo[0]
        if flavor[-2] == '1':
            flavorToPredict.append(int(flavor[-2:]))
        else:
            flavorToPredict.append(int(flavor[-1]))
    target = lines[flavors+4].strip("\n")
    startTime = datetime.datetime.strptime(lines[-2].strip("\n").split(" ")[0],'%Y-%m-%d')
    endTime =  datetime.datetime.strptime(lines[-1].strip("\n").split(" ")[0],'%Y-%m-%d')
    timeToPredict = (endTime - startTime).days + 1
    return physicalServer,flavorToPredict,flavorDict,target,timeToPredict

#print ReadInputData('input_5flavors_cpu_7days.txt')



