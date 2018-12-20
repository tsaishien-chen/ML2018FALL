from gensim.models import Word2Vec
import numpy as np
import csv
import sys
import jieba

if (len(sys.argv) != 5):
    print("#usage: python3 word2vec.py <embedding dim> <dict.txt.big file> <train_x file> <test_x.csv file>")

jieba.set_dictionary(sys.argv[2])
#seg_list = jieba.cut('在下雨的瞬間我多看了一眼')
#print(','.join(seg_list))

with open('negWords.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        jieba.add_word(data)
        
embedingDim = int(sys.argv[1])
train_Dict = []
test_Dict  = []

with open(sys.argv[3], newline='') as csvfile:
    rows = csv.reader(csvfile)
    next(rows, None)
    for row in rows:
        train_Dict += [[word for word in jieba.cut(row[1])]]
with open(sys.argv[4], newline='') as csvfile:
    rows = csv.reader(csvfile)
    next(rows, None)
    for row in rows:
        test_Dict += [[word for word in jieba.cut(row[1])]]    

model = Word2Vec(train_Dict+test_Dict, size=embedingDim, min_count=5, iter=10)
model.save('word2vec_'+str(embedingDim)+'.model')
