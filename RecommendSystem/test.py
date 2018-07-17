# encoding:utf-8

result_file = open("E:\\CCIR\\80-result\\result.csv", 'r', encoding='UTF-8')
two_result_file = open("E:\\CCIR\\85-result\\result.csv", 'r', encoding='UTF-8')

result_set = set()
two_result_set = set()
for line in result_file:
    temp_set = line.split(',')
    # print(len(temp_set))
    for i in temp_set:
        result_set.add(i)
print(f"80-result包含ID：{len(result_set)}")
del temp_set
for line in two_result_file:
    temp_set = line.split(',')
    for index in temp_set:
        two_result_set.add(index)
print(f"90-result包含ID：{len(two_result_set)}")
differe_set = two_result_set.difference(result_set)
print(f"差集大小{len(differe_set)}")
print(differe_set)
result_file.close()
two_result_file.close()

