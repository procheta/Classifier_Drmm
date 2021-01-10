import numpy as np
import sys

from transformers import AutoTokenizer, AutoModel,RobertaTokenizer, TFRobertaModel
from transformers import pipeline


def vec2str(x):
    vecstr = ''
    for x_i in x:
        x_i_str = '%.4f' %(x_i)
        vecstr += x_i_str + ' '

    return vecstr[0:-1]

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print ('usage: python saveptvecs.py <context file> <outvec file>')
        sys.exit(0)
    print("here")
    bertTokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    bertModel = TFRobertaModel.from_pretrained("roberta-base")

    vocabfile = sys.argv[1]
    outfile = sys.argv[2]

    nlp_features = pipeline('feature-extraction', model=bertModel, tokenizer=bertTokenizer)

    with open(vocabfile) as f:
        words = f.read().splitlines()

    f = open(outfile, "w")
    f.write(str(len(words)) + ' 768\n')

    for w in words:
        try:
            output = nlp_features(w)
            output = np.array(output)
            output = output[0]

            vec = output[1] # middle vector, first is bos, last is eos
            vecstr = vec2str(vec)

            f.write(w + '\t' + vecstr + '\n')
        except:
            print ("Extraction failed for text {}".format(w))

    f.close()
