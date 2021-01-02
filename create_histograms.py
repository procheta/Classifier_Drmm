import os
import sys
import numpy as np
from TextCollection import TextCollection



flag=0
wv_vocab={}
idWordMap={}
index=0
wordvec={}
maxNum=0
minNum=9999
with open(sys.argv[4],"r") as f:
    for line in f:
        if flag == 0:
            flag=1
        else:
            st = line.split("\n")[0]
            words = st.split(" ")
            word = words[0]
            wv_vocab[word]=index
            idWordMap[index]=word
            vec=[]
            flag1=0
            sumVec=0
            maxValue=0
            for word in words:
                if flag1==0:
                    flag1=1
                else:
                    try:
                        vec.append(float(word))
                        number = float(word)
                        if number < 0:
                            number = -number
                        if maxValue < number:
                            maxValue=number
                    except:
                        n=0
            for i in range(len(vec)):
                vec[i] = vec[i]/maxValue
                if maxNum < vec[i]:
                    maxNum = vec[i]
                if minNum > vec[i]:
                    minNum=vec[i]
            wordvec[index]=vec
            index=index+1

print(maxNum)
print(minNum)
# from matchzoo: https://github.com/faneshion/MatchZoo/blob/master/matchzoo/inputs/preprocess.py
def cal_hist(t1_rep, t2_rep, qnum, hist_size):

    mhist = np.zeros((qnum, hist_size), dtype=np.float32)
    mm = t1_rep.dot(np.transpose(t2_rep))


    for (i,j), v in np.ndenumerate(mm):
        if i >= qnum:
            break
        vid = int((v + 1.) /(hist_size - 1.))
        mhist[i][vid] += 1.

    mhist += 1.
    mhist = np.log10(mhist)
    return mhist.flatten()


# make sure the argument is good (0 = the python file, 1+ the actual argument)
if len(sys.argv) < 7:
    print('Needs 6 arguments - see comments for info ...')
    exit(0)

arg_corpus_file = sys.argv[1]
arg_topics_file = sys.argv[2]
arg_preranked_or_qrel_file = sys.argv[3]
arg_embedding_file = sys.argv[4]
arg_bin_size = int(sys.argv[5])
arg_qrel_or_preranked = sys.argv[6] # qrel or prerank

#
# load pre-ranked or qrels file
#
topic_doc_pairs = [] # (topic, doc id, score) score is 0,1 if qrel

if arg_qrel_or_preranked == 'prerank':
    with open(arg_preranked_or_qrel_file, 'r') as inputFile:
        for line in inputFile:
            parts = line.split("\t")
            topic_doc_pairs.append((parts[0], parts[1].strip(), parts[2].strip()))

if arg_qrel_or_preranked == 'qrel':
    with open(arg_preranked_or_qrel_file, 'r') as inputFile:
        for line in inputFile:
            parts = line.split()
            topic_doc_pairs.append((parts[0], parts[2].strip(), parts[3].strip()))

print('all ', len(topic_doc_pairs), ' topic_doc pairs loaded')

# load word embedding
#model = Word2Vec.load(arg_embedding_file)
#vectors = model.wv
#del model
#vectors.init_sims(True) # normalize the vectors (!), so we can use the dot product as similarity measure

print('embeddings loaded ')
print('loading docs ... ')

# load trec corpus
trec_text_collection_data = [] # text 1 string per doc only, no id
trec_corpus={} # corpus id -> list of doc vector ids
count = 0
with open(arg_corpus_file, 'r') as inputFile:
    for line in inputFile:
        count+=1
        if count % 100000==0:
            print('    ', count,' docs loaded')
        parts = line.split('\t', 1)
        trec_corpus[parts[0].strip()] = []
        trec_text_collection_data.append(parts[1])

        flag = 0
        for w in parts[1].split(' '):
            ws = w.strip()
            if ws in wv_vocab:
                trec_corpus[parts[0]].append(wv_vocab.get(ws))
                flag=1
        #if flag == 0:
            #print("No vector found ",parts[0])
trec_text_collection = TextCollection(trec_text_collection_data)

print('all ', count, ' docs loaded')
print(len(trec_corpus))
# load topics file
trec_topics = {} # topic -> list of query term vector ids
max_topic_word_count = 0
with open(arg_topics_file, 'r') as inputFile:
    for line in inputFile:
        parts = line.split('\t', 1)
        parts[0] = parts[0].strip()
        if parts[0] not in trec_topics:
            trec_topics[parts[0]] = []

        for w in parts[1].split(' '):
            ws = w.strip()
            if ws in wv_vocab:
                trec_topics[parts[0]].append(wv_vocab.get(ws))

        if len(trec_topics[parts[0]]) > max_topic_word_count:
            max_topic_word_count = len(trec_topics[parts[0]])

print('all ', len(trec_topics), ' topics loaded')

print('creating histograms')
count = 0
# create histograms for every query term <-> doc term
# based on pairs from pre-ranked file, using the similarities of the wordembedding

# histogram file format: topicId DocId prerankscore numberOfTopicWords(N) idf1 idf2 ... idfN <hist1> <hist2> ... <histN>
ll=[]
tt=[]
dd=[]
with open('/home/procheta/Histogramtriplefullqrel_'+str(arg_bin_size)+'.txt', 'w') as outputFile:


    for topic, doc, score in topic_doc_pairs:
            count += 1
            if count % 10000 == 0:
                print('    ', count, ' ranked docs processed')

            if doc not in trec_corpus:
                print('skipping doc (not in corpus): '+ doc)
                continue
            if topic not in trec_topics:
                print('skipping topic (not in corpus): ' + topic)
                continue

            # get the word embedding ids
            topic_word_ids = trec_topics[topic]
            doc_words_ids = trec_corpus[doc]

            
            topic_vectors = np.array([wordvec.get(topic_word_ids[i]) for i in range(0,len(topic_word_ids))],np.float32)
            doc_vectors = np.array([wordvec.get(doc_words_ids[i]) for i in range(0,len(doc_words_ids))],np.float32)

            qnum = len(topic_word_ids)
            d1_embed = topic_vectors
            d2_embed = doc_vectors

            try:
                curr_hist = cal_hist(d1_embed, d2_embed, qnum, arg_bin_size)
                curr_hist = curr_hist.tolist()
                outputFile.write(topic+" "+doc+" "+str(score)+" "+str(len(topic_word_ids))+ " ")
                ll.append(score)
                tt.append(topic)
                dd.append(doc)
                for w in topic_word_ids:
                    outputFile.write(str(trec_text_collection.idf(idWordMap.get(w)))+" ")
                outputFile.write(' '.join(map(str, curr_hist)))
                outputFile.write('\n')
                outputFile.flush()
            except Exception as e:
                outputFile.flush()
                print("Exception")
                #print(doc_words_ids)
                print(topic_word_ids)
                print(topic)
                #print(e)

                #sys.exit()
print(len(tt))
print(len(dd))
with open("/home/procheta/modifed_querypair_train.txt","w") as f:
    for i in range(len(tt)):
        f.write(tt[i]+" "+dd[i]+" "+ll[i])
        f.write("\n")

f.close()
print('Max topic words: ', max_topic_word_count)
