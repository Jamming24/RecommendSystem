# -*- coding: utf-8 -*-
# @Time    : 2019/5/7 10:22
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : 上网明细处理.py
# @Software: PyCharm

file_path = "C:\\Users\\Jamming\\Desktop\\7公寓-精简.csv"
out_path = "C:\\Users\\Jamming\\Desktop\\7公寓-精简_fliter.csv"
out_path2 = "C:\\Users\\Jamming\\Desktop\\7公寓-精简_fliter_2.csv"
file_object = open(file_path, 'r', encoding='UTF-8')
out_object = open(out_path, 'w', encoding='UTF-8')
out_object2 = open(out_path2, 'w', encoding='UTF-8')
flag = True
line_num = 0
data_dict = dict()

for line in file_object:
    line_num += 1
    if flag:
        flag = False
        continue
    else:
        values = line.split(",")
        account = values[0]
        VLAN_ID = values[4]
        keys = data_dict.keys()
        if VLAN_ID in keys:
            temp_list = data_dict[VLAN_ID]
            if account in temp_list:
                continue
            else:
                temp_list.append(account)
                data_dict[VLAN_ID] = temp_list
        else:
            account_list = list()
            account_list.append(account)
            data_dict[VLAN_ID] = account_list

print(f"总计{line_num}行")
sum_account = 0
out_object.write("公寓信息,VLAN ID,账号数量,账号列表\n")
for key in data_dict.keys():
    print(key+">>>>>")
    out_object.write("7公寓,"+key+",")
    print(data_dict[key])
    sum_account += len(data_dict[key])
    out_object.write(str(len(data_dict[key]))+",")
    for ac in data_dict[key]:
        out_object.write(ac+",")
    out_object.write("\n")
print(sum_account)
file_object.close()
out_object.close()
out_object2.write("异常次数,账号列表\n")
num_dict = dict()
for key in data_dict.keys():
    num = str(len(data_dict[key]))
    if num in num_dict.keys():
        temp = num_dict[num]
        temp.append(key)
    else:
        num_list = []
        num_list.append(key)
        num_dict[num] = num_list
for newKey in num_dict.keys():
    out_object2.write(newKey+",")
    out_object2.write(str(len(num_dict[newKey])) + ",")
    for i in num_dict[newKey]:
        out_object2.write(i+",")
    out_object2.write("\n")
out_object2.close()


