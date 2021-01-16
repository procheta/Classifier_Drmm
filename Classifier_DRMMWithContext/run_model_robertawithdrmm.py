import sys
import keras
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras_model_robertawithdrmm import build_keras_model
from load_data_robertawithdrmm import *
from loss_function import *
import numpy as np
import os


# make sure the argument is good (0 = the python file, 1+ the actual argument)
if len(sys.argv) < 10:
    print('Needs 5 arguments - 1. run name, 2. train pair file (fold), 3. train histogram file, 4. test file (fold), 5. test histogram file')
    exit(0)

run_name = sys.argv[1]
train_file = sys.argv[2]
train_file_histogram = sys.argv[3]
test_file = sys.argv[6]
test_file_histogram = sys.argv[7]

#
# build and train model
#
model = build_keras_model()
model.summary()
model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam') # adam

x_0=get_bert_vec(sys.argv[4])
x_1=get_bert_vec(sys.argv[5])

train_input, train_labels = get_keras_train_input(train_file, train_file_histogram)


if not os.path.exists('models/'):
    os.makedirs('models/')




model.fit([train_input,x_0,x_1], train_labels, batch_size=5000, verbose=1, shuffle=False, epochs=50)#, callbacks=[c1])


model.save_weights('models/'+run_name+'.weights')

#
# prediction
#

test_data,pre_rank_data = get_keras_test_input(test_file, test_file_histogram)
x_0=get_bert_vec(sys.argv[8])
x_1=get_bert_vec(sys.argv[9])

predictions = model.predict([test_data['doc'],x_0,x_1], batch_size = 10)

if not os.path.exists('result/'):
    os.makedirs('result/')
with open('result/'+run_name+".result1", 'w') as outFile:
    i = 0
    for topic, doc in pre_rank_data:
        outFile.write(topic + ' '+doc+' '+str(predictions[i][0])+'\n')
        i += 1
