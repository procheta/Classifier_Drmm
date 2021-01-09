import keras
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation,  Lambda, Permute, Dropout,Concatenate
from keras.layers import Reshape, Dot,Add, Lambda
from keras.regularizers import l2
from keras.activations import softmax

query_term_maxlen=25 
hist_size =10 
num_layers=1 
hidden_sizes = [3]
roberta_dimension=768
initializer_fc = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=11)
initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)


#
# returns the raw keras model object
#

def build_keras_model():

    #
    # input layers (query and doc)
    #

    # -> the query idf input (1d array of float32)
    #query = Input(name='query', shape=(query_term_maxlen,1))

    # -> the histogram (2d array: every query gets 1d histogram
    doc = Input(name='doc', shape=(hist_size,))
    vec1= Input(name='v1',shape=(roberta_dimension,))
    vec2=Input(name='v2',shape=(roberta_dimension,))
    #
    # the histogram handling part (feed forward network)
    #
    processed_vec1=Dense(100,activation='selu')(vec1)
    processed_vec2=Dense(100,activation='selu')(vec2)
    
    processed_vec11=Dropout(0.2)(processed_vec1)
    processed_vec22=Dropout(0.2)(processed_vec2)
    mid=Concatenate(axis=1)([processed_vec11, processed_vec22])

    z2=Dense(1,activation='sigmoid')(mid)



    model = Model(inputs=[doc,vec1,vec2], outputs=[z2])

    return model
