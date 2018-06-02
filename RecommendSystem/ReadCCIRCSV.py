# coding=utf-8

import csv
import gc

def readSample():
    with open('E:\\BaiduNetdiskDownload\\CCIR\\sample\\result.csv','r') as myFile:
        lines  =csv.reader(myFile)
        for li in lines:
            print(li)


def StatisticsCandidateSet():
    count = 0
    answerCount = 0
    quesionCount = 0
    file_object = open("E:\\BaiduNetdiskDownload\\CCIR\\competition\\candidate.txt", 'r')
    contents = file_object.readlines()
    for line in contents:
        count += 1
        data = line.split("\t")
        if data[0] == 'A':
            answerCount += 1
        else:
            quesionCount += 1
    print(f'答案数量为{answerCount}，，，问题数量为：{quesionCount}')
    print("行数 ：", count)


# StaticalCandidateSet()

def StatisticsTesting():
    allCount = 0
    file_object = open("E:\\BaiduNetdiskDownload\\CCIR\\competition\\testing_set.txt", 'r', encoding='UTF-8')
    allContent = file_object.readlines()
    for every in allContent:
        allCount += 1
        # print(every)
    print(allCount)


def StatisticsTrainSet():
    file_object = open("E:\\BaiduNetdiskDownload\\CCIR\\competition\\training_set.txt", 'r', encoding="UTF-8")
    OutFile_object = open("E:\\BaiduNetdiskDownload\\CCIR\\competition\\中间数据\\StatisticUserBehavior.txt", 'w')
    dataline = file_object.readline()
    statisticsDict = dict()
    while dataline:
        temp = dataline.split()
        if len(temp) == 2:
            # userId = temp[0]
            frequency = temp[1]
            if frequency in statisticsDict.keys():
                userIdCount = statisticsDict[frequency]
                userIdCount += 1
                statisticsDict[frequency] = userIdCount
                gc.collect()
            else:
                statisticsDict[frequency] = 1
            dataline = file_object.readline()
        else:
            continue

    ct = 0
    for key in statisticsDict.keys():
        OutFile_object.writelines(str(key) + "," + str(statisticsDict[key]) + '\n')
        ct += statisticsDict[key]
    print(f'总计行为数据{ct}条')

    file_object.close()
    OutFile_object.close()

    return statisticsDict


StatisticsTrainSet()
