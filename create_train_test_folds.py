#
# Generate 5-fold split & 1-0 pairs for training and 5-fold split for testing
#
# Input:  qrels file (for training data), pre-ranked file (for test data) -> should be from the same test collection
# Output: (fixed) ../data/5-folds/<qrel-file-name>_fold_n.train
#                 ../data/5-folds/<qrel-file-name>_fold_n.test
#
# the split is created by the topic ids

import sys
import os
from random import randint
import traceback

# make sure the argument is good (0 = the python file, 1 the actual argument)

input_file = sys.argv[1]

#
# load qrels file
# ---------------------------------------------------------
#
qrels_doc_pairs = {} # topic -> (relevant: [docId], non-relevant: [docId]) (, doc id, relevant) relevant is boolean

count_rel = 0
count_non_rel = 0
qrel_name=os.path.basename(input_file).replace('.txt','')



inputs=[]
with open(input_file, 'r') as inputFile:
    for line in inputFile:
        parts = line.split()
        x=[]
        x.append(parts[0])
        x.append(parts[1])
        x.append(parts[2])
        inputs.append(x)

print(len(inputs), ' inputs loaded')


# here we could shuffle things around

part_size = int(round(len(inputs)/5,0))
part_1 = inputs[:part_size]
part_2 = inputs[part_size:2*part_size]
part_3 = inputs[2*part_size:3*part_size]
part_4 = inputs[3*part_size:4*part_size]
part_5 = inputs[4*part_size:]

print('part sizes 1-5:',len(part_1),len(part_2),len(part_3),len(part_4),len(part_5))
#print('parts:\n[0] ',' '.join(str(part_1)),'\n[1] ',' '.join(part_2),'\n[2] ',' '.join(part_3),'\n[3] ',' '.join(part_4),'\n[4] ',' '.join(part_5))

#
# load pre-ranked
# ---------------------------------------------------------
#

#
# load pre-ranked file
#

#print(count_prerank, ' pre-ranked docs loaded')

#
# output helper functions
# ---------------------------------------------------------

# generate output per topic
# combine pairs for every rel - every non_rel
def create_1_0_pairs(topics):
    lines = []

    for i in range(len(topics)):
        x=topics[i]
        lines.append(x[0]+" "+x[1]+" "+x[2]+"\n")
    print('\t got  ',len(lines),'train pairs')

    return lines

# combine the parts
def writeOutFiles(train1,train2,train3,train4,test,foldnumber):

    print('saving fold '+str(foldnumber))

    with open('../data/5-folds/'+str(qrel_name)+'_fold_'+str(foldnumber)+'.train', 'w') as trainFile:
        trainFile.writelines(create_1_0_pairs(train1))
        trainFile.writelines(create_1_0_pairs(train2))
        trainFile.writelines(create_1_0_pairs(train3))
        trainFile.writelines(create_1_0_pairs(train4))

    with open('../data/5-folds/' + str(qrel_name) + '_fold_' + str(foldnumber) + '.test', 'w') as testFile:
        lines = []
        for topic in test:
            lines.append(topic[0] + ' ' + topic[1]+'\n')

        print('\t got  ', len(lines), 'test docs')
        testFile.writelines(lines)

#
# create the folds
# ---------------------------------------------------------

if not os.path.exists('../data/5-folds/'):
    os.makedirs('../data/5-folds/')

# train 1,2,3,4, test 5
writeOutFiles(part_1,part_2,part_3,part_4,part_5, 1)

# train 1,2,3,5, test 4
writeOutFiles(part_1,part_2,part_3,part_5,part_4, 2)

# train 1,2,4,5, test 3
writeOutFiles(part_1,part_2,part_4,part_5,part_3, 3)

# train 1,3,4,5, test 2
writeOutFiles(part_1,part_3,part_4,part_5,part_2, 4)

# train 2,3,4,5, test 1
writeOutFiles(part_2,part_3,part_4,part_5,part_1, 5)
