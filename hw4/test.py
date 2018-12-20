import sys
import csv
import gensim
import numpy as np
import jieba
    
import keras
from keras.preprocessing.sequence import pad_sequences

if (len(sys.argv) != 8):
    print("#usage: python3 <testing code> <test_x file> <dict.txt.big file> <output file> <RNN word2vec model> <RNN training model> <DNN word2vec model> <DNN training model>")

jieba.set_dictionary(sys.argv[2])
#seg_list = jieba.cut('在下雨的瞬間我多看了一眼')
#print(','.join(seg_list))

# <RNN> Data Reading and Preprocessing
word2vec_model = gensim.models.Word2Vec.load(sys.argv[4])
test_X_rnn = []   
with open(sys.argv[1], newline='') as csvfile:
    rows = csv.reader(csvfile)
    next(rows, None)
    for row in rows:
        test_X_rnn.append([])
        seg = jieba.cut(row[1])
        for word in seg:
            if word in word2vec_model.wv.vocab : test_X_rnn[-1].append(word2vec_model[word])

maxlength = 48
test_X_rnn = pad_sequences(test_X_rnn,
                           maxlen = maxlength,
                           padding = 'post',
                           truncating = 'post',
                           value = word2vec_model[' '])

model_rnn = keras.models.load_model(sys.argv[5])
test_Y_rnn = model_rnn.predict(test_X_rnn).flatten()

# <DNN> Data Reading and Preprocessing      
word2vec_model = gensim.models.Word2Vec.load(sys.argv[6])
numWords = len(word2vec_model.wv.vocab)
test_X_dnn = np.zeros((len(test_X_rnn), numWords))
with open(sys.argv[1], newline='') as csvfile:
    rows = csv.reader(csvfile)
    next(rows, None)
    counter = 0
    for row in rows:               
        seg = jieba.cut(row[1])
        for word in seg:
            if word in word2vec_model.wv.vocab:
                test_X_dnn[counter,word2vec_model.wv.vocab[word].index] += 1            
        counter += 1

model_dnn = keras.models.load_model(sys.argv[7])
test_Y_dnn = model_dnn.predict(test_X_dnn).flatten()

# Combine the predictions from two models
test_Y = test_Y_rnn+test_Y_dnn*1.6
threshold = 1.3
test_Y[test_Y <= threshold] = 0
test_Y[test_Y >  threshold] = 1

with open(sys.argv[3], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','label'])
    for i in range(test_Y.size):
        writer.writerow([i, int(test_Y[i])])
