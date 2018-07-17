# encoding:utf-8


def processErrorline():
    file = "E:\\CCIR\\available_topic_info.txt"
    new_file = "E:\\CCIR\\2_available_topic_info.txt"
    file_object = open(file, 'r', encoding='UTF-8')
    Out_file_object = open(new_file, 'w', encoding='UTF-8')
    count = 0
    text_list = []
    for line in file_object:
        count += 1
        if len(line.split('\t')[0]) != 32:
            temp_line = text_list[len(text_list)-1].strip("\n")
            text_list.pop()
            temp_line += line.strip("\n")
            text_list.append(temp_line+"\n")
        else:
            text_list.append(line)
    file_object.close()
    for li in text_list:
        Out_file_object.write(li)
    Out_file_object.close()


def get_Candidate_question_infos():
    file = "E:\\CCIR\\candidate_answer.txt"
    file_object = open(file, 'r', encoding='UTF-8')
    count = 0
    question = []
    for line in file_object:
        question.append(line.split('\t')[1])

    print(len(set(question)))


