# encoding:utf-8


def load_ItemCF_ReCsv(item_result_withScore_file):
    ItemCFRec_dict = dict()
    saveItemDic = dict()
    file_object = open(item_result_withScore_file, 'r', encoding='UTF-8')
    for line in file_object:
        simple_ItemCFRec_dic = dict()
        save_ItemCF = []
        t = line.split('\t')
        user_ID = t[0]
        reclist = t[1].strip('\n').split(',')
        reclist.remove('')
        index = 0
        for recitem in reclist:
            it = recitem.split(':')
            if index < 5:
                save_ItemCF.append(it[0])
            else:
                simple_ItemCFRec_dic[it[0]] = float(it[1])
            index += 1
        saveItemDic[user_ID] = save_ItemCF
        ItemCFRec_dict[user_ID] = simple_ItemCFRec_dic
    file_object.close()
    print(f"相似度字典大小{len(ItemCFRec_dict)}")
    return ItemCFRec_dict, saveItemDic


def load_Popular_Candidate(popular_candidate_file):
    # 短ID流行度候选池
    Popular_candidate = dict()
    file_object = open(popular_candidate_file, 'r', encoding='UTF-8')
    for line in file_object:
        tt = line.strip('\n').split(',')
        Popular_candidate[tt[0]] = float(tt[1])
    print(f"候选集数量：{len(Popular_candidate)}")
    return Popular_candidate


def load_revers_answer_dict(answer_id_dict_file):
    # 根据长ID找短ID
    revers_answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        revers_answer_id_dict[line.split('\t')[1][:32]] = line.split('\t')[0]
    return revers_answer_id_dict


def popular_Item_Mix(Popular_candidate, ItemCFRec_dict, revers_answer_id_dict):
    popular_Item_Mix_Dict = dict()
    for userID in ItemCFRec_dict.keys():
        simply_popular_Item_Mix = dict()
        simply_ItemCF_dic = ItemCFRec_dict[userID]
        for item in simply_ItemCF_dic.keys():
            shortID = 'A' + revers_answer_id_dict[item]
            if shortID in Popular_candidate.keys():
                Item_score = simply_ItemCF_dic[item]
                popular_score = Popular_candidate[shortID]
                mix_score = Item_score + popular_score
                simply_popular_Item_Mix[item] = mix_score
            else:
                simply_popular_Item_Mix[item] = simply_ItemCF_dic[item]
        popular_Item_Mix_Dict[userID] = simply_popular_Item_Mix
    print(f"混合推荐用户数量{len(popular_Item_Mix_Dict)}")
    return popular_Item_Mix_Dict


def Print_Popular_Item_Mix(saveItemDic, popular_Item_Mix_Dict, outfile):
    file_object = open(outfile, 'w', encoding='UTF-8')
    for userID in saveItemDic.keys():
        Item_Mix_sort = sorted(popular_Item_Mix_Dict[userID].items(), key=lambda e: e[1], reverse=True)
        file_object.write(userID + '\t')
        index = 0
        for item in saveItemDic[userID]:
            index += 1
            file_object.write(item + "@" + str(index) + ',')
        print(">>>>>>>>>>>>>>>>>>>>>")
        for mixItem in Item_Mix_sort:
            if index < 10:
                index += 1
                # print(mixItem[0] + ',' + str(mixItem[1]))
                file_object.write(mixItem[0] + "@" + str(index) + ',')
        print(">>>>>>>>>>>>>>>>>>>>>>")
        file_object.write('\n')
    file_object.close()
    print("混合推荐输出完成")


if __name__ == '__main__':
    popular_candidate_file = "F:\\CCIR_online\\popular\\popular_25,26,27,28,29,30,31.txt"
    item_result_withScore_file = "F:\\CCIR_online\\20180801\\中间文件\\item_result_withScore_file_20180731.csv"
    answer_id_dict_file = "F:\\CCIR_online\\20180801\\answer_id_all.dict"
    outfile = "F:\\CCIR_online\\20180801\\中间文件\\itemCF_popular_mix.csv"
    Popular_candidate = load_Popular_Candidate(popular_candidate_file)
    ItemCFRec_dict, saveItemDic = load_ItemCF_ReCsv(item_result_withScore_file)
    revers_answer_id_dict = load_revers_answer_dict(answer_id_dict_file)
    popular_Item_Mix_Dict = popular_Item_Mix(Popular_candidate, ItemCFRec_dict, revers_answer_id_dict)
    Print_Popular_Item_Mix(saveItemDic, popular_Item_Mix_Dict, outfile)
