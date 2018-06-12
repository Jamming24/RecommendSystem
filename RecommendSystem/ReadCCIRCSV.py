# coding=utf-8

import csv
import gc
import os


def readSample():
    with open('E:\\BaiduNetdiskDownload\\CCIR\\sample\\result.csv', 'r') as myFile:
        lines = csv.reader(myFile)
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



def StatisticsTesting():
    allCount = 0
    file_object = open("E:\\BaiduNetdiskDownload\\CCIR\\competition\\testing_set.txt", 'r', encoding='UTF-8')
    allContent = file_object.readlines()
    for every in allContent:
        allCount += 1
        # print(every)
    print(allCount)


def StatisticsTrainSet():
    # 此程序无法完成指定任务，不要使用
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


def statisticsUserBehavior():
    # 统计用户行为信息：统计每个用户用于多少条行为信息，对高频用户进行筛选，顺便将冷启动用户进行筛选
    # 且样本中行为数（交互 + 搜索词）小于 20 的样本」为冷启动用户样本

    # 代码逻辑  将每个文件读取到内存中,按照用户ID为key,用户行为数量为value
    testfloder = "E:\\CCIR测试数据"
    outfile = "E:\\CCIR测试数据\\user_behavior_statistics.txt"
    Files = os.listdir(testfloder)
    user_behavior_dic = dict()
    allcount = 0
    dicount = 0

    # 统计每个用户有几条用户行为信息
    for file in Files:
        if len(file) > 18:
            simple_file_path = os.path.join(testfloder, file)
            file_object = open(simple_file_path, 'r', encoding='UTF-8')
            data = file_object.readlines()
            for line in data:
                allcount += 1
                temp = line.split()
                try:
                    id = temp[0]
                except Exception:
                    print("发生数组下标越界")
                    continue
                else:
                    if id in user_behavior_dic.keys():
                        value = user_behavior_dic[temp[0]]
                        value += 1
                        user_behavior_dic[temp[0]] = value
                    else:
                        user_behavior_dic[temp[0]] = 1


            file_object.close()
        else:
            continue
        print(file, "读取完成")
    # print(user_behavior_dic)
    print(f"数据行：{allcount}")

    out_file_object = open(outfile, 'w', encoding='UTF-8')
    for key in user_behavior_dic.keys():
        out_file_object.write(key+","+str(user_behavior_dic[key])+"\n")
        dicount += user_behavior_dic[key]
    out_file_object.close()

    print(f"总计：{dicount}行")


file_object = open("E:\\CCIR测试数据\\user_behavior_statistics.txt", 'r', encoding="UTF-8")
data = file_object.readlines()
all_count = 0
count = 0
error_count = 0
max_count = 0
for line in data:
    all_count += 1
    temp = line.split(",")
    try:
        n = int(temp[1])
    except Exception:
        print("格式出错，忽略")
        error_count += 1
        continue
    else:
        if n == 172:
            count += 1

print(f"总行数:{all_count}")
print(f"最多{count}个")
print(f"出错行数：{error_count}")


