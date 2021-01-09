import keras
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation,  Lambda, Permute
from keras.layers import Reshape, Dot,Add, Lambda
from keras.activations import softmax

query_term_maxlen=25 
hist_size =5 
num_layers=1 
hidden_sizes = [3]

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

    #
    # the histogram handling part (feed forward network)
    #


    z = Dense(hidden_sizes[0], kernel_initializer=initializer_fc)(doc)
    z1=Dense(1,activation='sigmoid')(z)


    #q_w = Dense(1, kernel_initializer=initializer_gate, use_bias=False)(query) # what is that doing here ??
    #q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(query_term_maxlen,))(q_w)
    #q_w = Reshape((query_term_maxlen,))(q_w) # isn't that redundant ??

    #
    # combination of softmax(query term idf) and feed forward result per query term
    #
    #out_ = Dot(axes=[1, 1])([z, q_w])

    model = Model(inputs=[doc], outputs=[z1])

    return model
