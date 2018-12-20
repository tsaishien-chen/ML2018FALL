import sys
import csv
import gensim
import numpy as np
import jieba
    
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, BatchNormalization
from keras.optimizers import Adam , SGD
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

if (len(sys.argv) != 5):
    print("#usage: python3 <training code> <train_x file> <train_y file> <dict.txt.big file> <word2vec model>")

jieba.set_dictionary(sys.argv[3])
#seg_list = jieba.cut('在下雨的瞬間我多看了一眼')
#print(','.join(seg_list))

f = sys.argv[4]
embeddingDim = int(f[f.find("_")+1:f.find(".")])

# Data Reading and Preprocessing
train_X = []
word2vec_model = gensim.models.Word2Vec.load(sys.argv[4])
with open(sys.argv[1], newline='') as csvfile:
    rows = csv.reader(csvfile)
    next(rows, None)
    for row in rows:
        train_X.append([])
        seg = jieba.cut(row[1])

        for word in seg:
            if word in word2vec_model.wv.vocab : train_X[-1].append(word2vec_model[word])

maxlength = 48
train_X = pad_sequences(train_X,
                       maxlen = maxlength,
                       padding = 'post',
                       truncating = 'post',
                       value = word2vec_model[' '])

train_Y = np.genfromtxt(sys.argv[2], delimiter=',')
train_Y = train_Y[1:,1:]

model =  Sequential()

modelName = "model_rnn"
model.add(LSTM(256,
               return_sequences = True,
               input_length = maxlength,
               input_dim = 256,
               dropout = 0.3,
               recurrent_dropout = 0.3,
               kernel_initializer='he_normal'))
model.add(LSTM(256,
               return_sequences = False,
               input_length = maxlength,
               input_dim = 256,
               dropout = 0.4,
               recurrent_dropout = 0.4,
               kernel_initializer='he_normal'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Model Compiling
adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# Callbacks
callbacks = []
modelcheckpoint = ModelCheckpoint(modelName+'_record/weights.{epoch:03d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True)
callbacks.append(modelcheckpoint)
csv_logger = CSVLogger(modelName+'_record/rnn_log.csv', separator=',', append=False)
callbacks.append(csv_logger)


training = model.fit(x = train_X,
                     y = train_Y,
                     validation_split = 0.2,
                     callbacks = callbacks,
                     epochs = 100,
                     batch_size = 64,
                     shuffle = True)

