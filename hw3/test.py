import numpy as np
import pandas as pd
import keras
import csv
import sys

df = pd.read_csv(sys.argv[1])
imagesize = len(str(df.feature[0]).split())
width     = int(imagesize**0.5)
height    = int(imagesize**0.5)
testsize  = df.shape[0]
test_X    = np.empty((testsize, height, width,1))

for i in range(testsize):
    test_X[i] = np.array(str(df.feature[i]).split()).reshape((height,width,1))
test_X /= 255

num_classes = 7

model = keras.models.load_model('model.h5')
test_Y = model.predict(test_X)
test_Y = np.argmax(test_Y,axis=1)

with open(sys.argv[2], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','label'])
    for i in range(test_Y.size):
        writer.writerow([i, test_Y[i]])
