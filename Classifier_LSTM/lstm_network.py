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
from sklearn.model_selection import KFold
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Embedding, LSTM, Bidirectional, Concatenate
from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras.optimizers import RMSprop
import sys
from keras.layers.merge import concatenate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

LSTM_DIM = 32
DROPOUT = 0.2
MAXLEN = 50
SEED=314159 # first digits of Pi... an elegant seed!
MODEL_FILE='trust_query_pairs.h5'
VOCAB_FILE='wiki2013-analyzed.vec'
RESULT_FILE="result.txt"

df = pd.read_csv(sys.argv[1])
df_query_1 = np.array(df["query"]) 
df_query_2 = np.array(df["variant"])
x_train = np.vstack([df_query_1, df_query_2])
x_train = np.transpose(x_train)
x_train1=np.transpose(df_query_1)
x_train2=np.transpose(df_query_2)


nltk.download('punkt')
corpora = []
corpora = df['query'].tolist()
corpora += df['variant'].tolist()


word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(corpora)
vocab_length = len(word_tokenizer.word_index) + 1

print("Total number of Words ",vocab_length)


word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(corpora, key=word_count)
max_len = len(word_tokenize(longest_sentence))
print("Maximum query length ",max_len)


query_1 = word_tokenizer.texts_to_sequences(df_query_1)
query_2 = word_tokenizer.texts_to_sequences(df_query_2)

query_1 = pad_sequences(query_1, max_len, padding='post')
query_2 = pad_sequences(query_2, max_len, padding='post')

x_train = np.hstack([query_1, query_2])
y_train = np.array(df["clicked"])
y_train = np.where(y_train < 1 , y_train, 1)




"""**Load Pre Trained word embedding** """

path_to_glove_file =VOCAB_FILE 
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
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


"""**Process the model**"""

def complete_model():
    
    input_a = Input(shape=(max_len, ))    
    
    emb_a = Embedding(embedding_matrix.shape[0],
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix])(input_a)
    
    input_b = Input(shape=(max_len, ))    
    
    emb_b = Embedding(input_dim=embedding_matrix.shape[0],
                  output_dim=embedding_matrix.shape[1],
                  weights=[embedding_matrix])(input_b)
    
    shared_lstm = LSTM(LSTM_DIM)

    processed_a = shared_lstm(emb_a)
    processed_a = Dropout(DROPOUT)(processed_a)
    processed_b = shared_lstm(emb_b)
    processed_b = Dropout(DROPOUT)(processed_b)

    merged_vector = concatenate([processed_a, processed_b], axis=-1)
    predictions = Dense(1, activation='sigmoid')(merged_vector)
    
    model = Model([input_a, input_b], outputs=predictions)
    return model


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



def trainModel(model,y_train, q1,q2):
    EPOCHS = 10
    BATCH_SIZE = 128
    history = model.fit([q1, q2], y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              #validation_data=([x_test[:, max_len], x_test[:, max_len: 2*max_len]], y_test),
              verbose=True
             )

    model.save_weights(MODEL_FILE)
    return history


def buildModel():
    model=complete_model()
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
    return model




kf = KFold(n_splits = 5, random_state=None, shuffle=False) # TODO : make shuffle = True 
total_accuracy = 0
total_precision = 0
total_f1 = 0
total_recall = 0

for train_index, test_index in kf.split(x_train):
  model = buildModel()
  model.summary()
  
  X_train, X_test = x_train[train_index], x_train[test_index]
  Y_train, Y_test = y_train[train_index], y_train[test_index]

  q1_train=query_1[train_index]
  q2_train=query_2[train_index]
  q1_test=query_1[test_index]
  q1_test_text=df_query_1[test_index]
  q2_test_text=df_query_2[test_index]
  q2_test=query_2[test_index]

  trainModel(model, Y_train,q1_train,q2_train)
  loss, accuracy = model.evaluate([q1_test, q2_test], Y_test, verbose=0)
  y_pred = model.predict([q1_test, q2_test])
  with open(RESULT_FILE,'a+') as f:
      for k in range(len(q1_test)):
        f.write(str(q1_test_text[k]))
        f.write("\t")
        f.write(str(q2_test_text[k]))
        f.write("\t")
        f.write(str(y_pred[k][0]))
        f.write("\n")
  f.close()

  print(y_pred.round())
  precision, recall, fscore, support = precision_recall_fscore_support(Y_test, y_pred.round(), average='macro')
  print('y_pred: ', y_pred.shape)
  print('loss: ', loss)
  print('accuracy: ', accuracy)
  print('f1:', fscore)
  print('precision: ', precision)
  print('recall: ', recall)
  total_accuracy += accuracy
  total_precision += precision
  total_recall += recall
  total_f1 += fscore
  K.clear_session()
  del model 

print('Avg accuracy: ', total_accuracy/5)
print('Avg f1: ', total_f1/5)
print('Avg precision: ', total_precision/5)
print('Avg recall: ', total_recall/5)
