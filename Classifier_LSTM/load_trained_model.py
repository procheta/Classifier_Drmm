from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import random as r
import math as m
import numpy as np
from keras import backend as K
from random import Random
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import nltk
import keras.preprocessing.text
from nltk.tokenize import word_tokenize
from numpy import newaxis
MAXLEN = 50
SEED=314159 # first digits of Pi... an elegant seed!
MODEL_FILE= 'trust_query_pairs1.h5'
import sys



df = pd.read_csv(sys.argv[1])
df1=pd.read_csv(sys.argv[2])
df_query_1 = np.array(df["query"]) 
df_query_2 = np.array(df["variant"])
df1_query_1=np.array(df1["query1"])
df2_query_2=np.array(df1["query2"])
x_train1=np.transpose(df_query_1)
x_train2=np.transpose(df_query_2)
Y_test=np.array(df1["yLabel"])


nltk.download('punkt')
corpora = []
corpora = df['query'].tolist()
corpora += df['variant'].tolist()
corpora+=df1["query1"].tolist()
corpora +=df1["query2"].tolist()

# print(corpora) 

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(corpora)
vocab_length = len(word_tokenizer.word_index) + 1

print(vocab_length)


word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(corpora, key=word_count)
max_len = len(word_tokenize(longest_sentence))
print(max_len)





query_test_1=np.array(df1["query1"])
query_test_2=np.array(df1["query2"])

q1_test_1=word_tokenizer.texts_to_sequences(query_test_1)
q2_test_2=word_tokenizer.texts_to_sequences(query_test_2)

q1_test_1 = pad_sequences(q1_test_1, max_len, padding='post')
q2_test_2 = pad_sequences(q2_test_2, max_len, padding='post')




path_to_glove_file = '/home/procheta/wiki2013-analyzed.vec'
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = vocab_length + 2
embedding_dim =200 
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


"""**Process the model**"""

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Embedding, LSTM, Bidirectional, Concatenate
from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras.optimizers import RMSprop

LSTM_DIM = 32
#LSTM_DIM = 48
DROPOUT = 0.2

from keras.layers.merge import concatenate

def complete_model():
    
    input_a = Input(shape=(max_len, ))    
    print (input_a.shape)
    
    emb_a = Embedding(embedding_matrix.shape[0],
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix])(input_a)
    print (emb_a.shape)
    
    input_b = Input(shape=(max_len, ))    
    print (input_b.shape)
    
    emb_b = Embedding(input_dim=embedding_matrix.shape[0],
                  output_dim=embedding_matrix.shape[1],
                  weights=[embedding_matrix])(input_b)
    print (emb_b.shape)
    
    shared_lstm = LSTM(LSTM_DIM)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = shared_lstm(emb_a)
    processed_a = Dropout(DROPOUT)(processed_a)
    processed_b = shared_lstm(emb_b)
    processed_b = Dropout(DROPOUT)(processed_b)

    merged_vector = concatenate([processed_a, processed_b], axis=-1)
    # And add a logistic regression (2 class - sigmoid) on top
    # used for backpropagating from the (pred, true) labels
    predictions = Dense(1, activation='sigmoid')(merged_vector)
    
    model = Model([input_a, input_b], outputs=predictions)
    return model

from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support

## quick hack:

def precsion_score(y_true, y_pred):
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return precision

def recall_score(y_true, y_pred):
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return recall

def fscore_value(y_true, y_pred):
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return fscore


#precision_recall_fscore_support(y_true, y_pred, average='macro')

def trainModel(model,y_train, q1,q2):
    #EPOCHS = 1
    EPOCHS = 1
    #BATCH_SIZE = 1000
    BATCH_SIZE = 128
    history = model.fit([q1, q2], y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              # validation_data=([x_val[:, max_len], x_val[:, max_len: 2*max_len]], y_val),
              #validation_data=([x_test[:, max_len], x_test[:, max_len: 2*max_len]], y_test),
              verbose=True
             )

    model.save_weights(MODEL_FILE)
    return history


def buildModel():
    model=complete_model()
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
    return model







total_accuracy = 0
total_precision = 0
total_f1 = 0
total_recall = 0

model = buildModel()
model.summary()
  


q1_test=q1_test_1
q2_test=q2_test_2
q1_test_text=query_test_1
q2_test_text=query_test_2

print(len(q1_test))
model.load_weights(MODEL_FILE)
loss, accuracy = model.evaluate([q1_test, q2_test], Y_test, verbose=0)
y_pred = model.predict([q1_test, q2_test])
with open("/home/procheta/result.txt",'w') as f:
    for k in range(len(q1_test)):
        f.write(str(q1_test_text[k]))
        f.write("\t")
        f.write(str(q2_test_text[k]))
        f.write("\t")
        f.write(str(y_pred[k][0]))
        f.write("\n")
f.close()








