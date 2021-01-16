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
import random
from random import randint
import traceback

# make sure the argument is good (0 = the python file, 1 the actual argument)
if len(sys.argv) < 4:
    print('Needs 2 arguments - 1. the input file, 2. left vector file,  3. right vector file')
    exit(0)
qrel_name=sys.argv[1]
input_file = sys.argv[2]
left_vec_file=sys.argv[3]
right_vec_file=sys.argv[4]
#
# load qrels file
# ---------------------------------------------------------
#

count_rel = 0
count_non_rel = 0



inputs=[]
with open(input_file, 'r') as inputFile:
    for line in inputFile:
        parts = line.split()
        x=[]
        x.append(parts[0])
        x.append(parts[1])
        x.append(parts[2])
        inputs.append(x)


left_vec=[]
with open(left_vec_file,'r') as vecFile:
    for line in vecFile:
        parts=line.split("\t")
        if (len(parts) > 1):
            left_vec.append(parts[1])

right_vec=[]
with open(right_vec_file, 'r') as vecFile:
    for line in vecFile:
        parts=line.split("\t")
        if (len(parts) > 1):
            right_vec.append(parts[1])

#random.shuffle(inputs)
# here we could shuffle things around

part_size = int(round(len(inputs)/5,0))
part_1 = inputs[:part_size]
part_2 = inputs[part_size:2*part_size]
part_3 = inputs[2*part_size:3*part_size]
part_4 = inputs[3*part_size:4*part_size]
part_5 = inputs[4*part_size:]

left_vec_part_1=left_vec[:part_size]
left_vec_part_2=left_vec[part_size:2*part_size]
left_vec_part_3=left_vec[2*part_size:3*part_size]
left_vec_part_4=left_vec[3*part_size:4*part_size]
left_vec_part_5=left_vec[4*part_size:]

right_vec_part_1=right_vec[:part_size]
right_vec_part_2=right_vec[part_size:2*part_size]
right_vec_part_3=right_vec[2*part_size:3*part_size]
right_vec_part_4=right_vec[3*part_size:4*part_size]
right_vec_part_5=right_vec[4*part_size:]





print('part sizes 1-5:',len(part_1),len(part_2),len(part_3),len(part_4),len(part_5))


def create_1_0_pairs(topics):
    lines = []

    for i in range(len(topics)):
        x=topics[i]
        lines.append(x[0]+" "+x[1]+" "+x[2]+"\n")
    print('\t got  ',len(lines),'train pairs')

    return lines



# combine the parts
def writeOutFiles(train1,train2,train3,train4,test,foldnumber,left_vecs1,left_vecs2,left_vecs3,left_vecs4,left_test_vecs,right_vecs1,right_vecs2, right_vecs3, right_vecs4,right_test_vecs):

    print('saving fold '+str(foldnumber))

    with open('../data/5-folds/'+str(qrel_name)+'_fold_'+str(foldnumber)+'.train', 'w') as trainFile:
        trainFile.writelines(create_1_0_pairs(train1))
        trainFile.writelines(create_1_0_pairs(train2))
        trainFile.writelines(create_1_0_pairs(train3))
        trainFile.writelines(create_1_0_pairs(train4))

    with open('../data/5-folds/'+str(qrel_name)+'_leftvec_fold_'+str(foldnumber)+'.train','w') as vecFile:
        vecFile.writelines(left_vecs1)
        vecFile.writelines(left_vecs2)
        vecFile.writelines(left_vecs3)
        vecFile.writelines(left_vecs4)
    
    with open('../data/5-folds/'+str(qrel_name)+'_rightvec_fold_'+str(foldnumber)+'.train','w') as vecFile:
        vecFile.writelines(right_vecs1)
        vecFile.writelines(right_vecs2)
        vecFile.writelines(right_vecs3)
        vecFile.writelines(right_vecs4)



    with open('../data/5-folds/' + str(qrel_name) + '_fold_' + str(foldnumber) + '.test', 'w') as testFile:
        lines = []
        for topic in test:
            lines.append(topic[0] + ' ' + topic[1]+'\n')


        print('\t got  ', len(lines), 'test docs')
        testFile.writelines(lines)
    with open("../data/5-folds/"+str(qrel_name)+'_leftvec_fold_'+str(foldnumber)+'.test','w') as testvecFile:
        testvecFile.writelines(left_test_vecs)
    with open("../data/5-folds/"+str(qrel_name)+'_rightvec_fold_'+str(foldnumber)+'.test','w') as testvecFile:
        testvecFile.writelines(right_test_vecs)

#
# create the folds
# ---------------------------------------------------------

if not os.path.exists('../data/5-folds/'):
    os.makedirs('../data/5-folds/')

# train 1,2,3,4, test 5
writeOutFiles(part_1,part_2,part_3,part_4,part_5, 1,left_vec_part_1,left_vec_part_2,left_vec_part_3,left_vec_part_4,left_vec_part_5,right_vec_part_1,right_vec_part_2,right_vec_part_3,right_vec_part_4,right_vec_part_5)


writeOutFiles(part_1,part_2,part_3,part_5,part_4, 2,left_vec_part_1,left_vec_part_2,left_vec_part_3,left_vec_part_5,left_vec_part_4,right_vec_part_1,right_vec_part_2,right_vec_part_3,right_vec_part_5,right_vec_part_4)

# train 1,2,3,5, test 4

# train 1,2,4,5, test 3
writeOutFiles(part_1,part_2,part_4,part_5,part_3, 3,left_vec_part_1,left_vec_part_2,left_vec_part_4,left_vec_part_5,left_vec_part_3,right_vec_part_1,right_vec_part_2,right_vec_part_4,right_vec_part_5,right_vec_part_3)

# train 1,3,4,5, test 2
writeOutFiles(part_1,part_3,part_4,part_5,part_2, 4,left_vec_part_1,left_vec_part_3,left_vec_part_4,left_vec_part_5,left_vec_part_2,right_vec_part_1,right_vec_part_3,right_vec_part_4,right_vec_part_5,right_vec_part_2)



# train 2,3,4,5, test 1
writeOutFiles(part_2,part_3,part_4,part_5,part_1, 5,left_vec_part_2,left_vec_part_3,left_vec_part_4,left_vec_part_5,left_vec_part_1,right_vec_part_2,right_vec_part_3,right_vec_part_4,right_vec_part_5,right_vec_part_1)

