# encoding:utf-8

import math

def getNewtest():
    flagPath = "E:\\CCIR\\flag.txt"
    test_file = "E:\\CCIR\\testing_set_Part.txt"
    small_test = "E:\\CCIR\\small_testing_set.txt"
    flag_list = []
    test_file_list = []
    Outlist = []
    flags_object = open(flagPath, 'r', encoding='UTF-8')
    test_file_object = open(test_file, 'r', encoding='UTF-8')
    Out_file_object = open(small_test, 'w', encoding='UTF-8')
    for line in test_file_object:
        test_file_list.append(line)

    for line in flags_object.readlines():
        flag_list.append(line[:1])
    print(f"flag标志：{len(flag_list)}")
    print(f"测试数据集数量{len(test_file_list)}行")
    for i in range(0, len(flag_list)):
        if flag_list[i] == '1':
            Outlist.append(test_file_list[i])

    flags_object.close()
    print(f"最后的小数据集数量:{len(Outlist)}行")

    for index in Outlist:
        Out_file_object.write(index)
    Out_file_object.close()


score = 1 / (1 + 1 / (math.exp(30/60)))
score2 = 1 / (1+1 / (math.exp(10/60)))
score3 = 1 / (1+1 / (math.exp(-(447/60))))
score4 = 1 / (1+1 / (math.exp(-(10/60))))
print('score:', score)
print('score2:', score2)
print('score3:', score3)
print('score4:', score4)

