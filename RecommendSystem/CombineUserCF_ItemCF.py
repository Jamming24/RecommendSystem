# encoding:utf-8

UserCF_csv = open("E:\\CCIR\\UserCF\\result.csv", 'r', encoding='UTF-8')
ItemCF_csv = open("E:\\CCIR\\ItemCF\\result.csv", 'r', encoding='UTF-8')
Combine_csv = open("E:\\CCIR\\Combine\\result.csv", 'w', encoding='UTF-8')
UserCF_list = []
ItemCF_list = []
Combine_list = []
for user in UserCF_csv:
    temp = user.split(',')[:50]
    UserCF_list.append(temp)
for Item in ItemCF_csv:
    temp = Item.split(',')[:50]
    ItemCF_list.append(temp)

print(len(UserCF_list))
print(len(ItemCF_list))
count = 0
all = 0
for i in range(0, len(UserCF_list)):
    count += 1
    temp_UserCF = UserCF_list[i]
    temp_ItemCF = ItemCF_list[i]
    temp_Combine = []
    temp_Combine = list(set(temp_ItemCF).union(set(temp_UserCF)))
    if len(temp_Combine) == 1:
        all += 1
        # print(count)
    # if len(temp_Combine) < 100:
    #     for k in range(0, 100 - len(temp_Combine)):
    #         temp_Combine.append('-1')
    # Combine_list.append(temp_Combine)
print(all)

# print(f"总行数{len(Combine_list)}")
# for index in range(0, len(Combine_list)):
#     temp_Combine = Combine_list[index]
#     print(f"单行长度{len(temp_Combine)}")
#     for item in temp_Combine:
#         Combine_csv.write(item+',')
#     Combine_csv.write('\n')


UserCF_csv.close()
ItemCF_csv.close()
Combine_csv.close()