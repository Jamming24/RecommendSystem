# encoding:utf-8


def loadTrainSet(train_file):
    
    file_object = open(train_file, 'r', encoding='UTF-8')
    for line in file_object:
        print(line.split('\t')[2])


train_file = "E:\\CCIR\\training_set_Part.txt"
loadTrainSet(train_file)
