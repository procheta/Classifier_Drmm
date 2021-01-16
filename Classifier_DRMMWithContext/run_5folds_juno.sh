#!/bin/bash

foldname="grissom_fold_"

trainFoldLocation="/home/procheta/data/5-folds/drmm_fold_"
trainHistogramLocation="Histogramorcasexpanded10_10.txt"
trainLeftVecLocation="/home/procheta/data/5-folds/drmm_leftvec_fold_"
trainRightVecLocation="/home/procheta/data/5-folds/drmm_rightvec_fold_"
testLeftVecLocation="/home/procheta/data/5-folds/drmm_leftvec_fold_"
testRightVecLocation="/home/procheta/data/5-folds/drmm_rightvec_fold_"
testFoldLocation="/home/procheta/data/5-folds/drmm_fold_"
testHistogramLocation="Histogramorcasexpanded10_10.txt"


for number in {1..5}
do
	#echo "running fold: $number"
	/home/procheta/spinning-storage/procheta/pythonNew/bin/python3 run_model_robertawithdrmm.py "$foldname$number" "$trainFoldLocation$number.train" "$trainHistogramLocation" "$trainLeftVecLocation$number".train "$trainRightVecLocation$number.train" "$testFoldLocation$number.test" "$testHistogramLocation" "$testLeftVecLocation$number.test" "$testRightVecLocation$number.test"
done
exit 0

#python3 run_model.py schirra_fold1 "../data/5-folds/robust04.qrels_fold_1.train" "../data/qrel_histogram_30.txt" "../data/5-folds/robust04.qrels_fold_1.test" "../data/prerank_histogram_30.txt" #&> logs/fold1.log &
#
#python3 run_model.py schirra_fold2 "../data/5-folds/robust04.qrels_fold_2.train" "../data/qrel_histogram_30.txt" "../data/5-folds/robust04.qrels_fold_2.test" "../data/prerank_histogram_30.txt" #&> logs/fold2.log &
#python3 run_model.py schirra_fold3 "../data/5-folds/robust04.qrels_fold_3.train" "../data/qrel_histogram_30.txt" "../data/5-folds/robust04.qrels_fold_3.test" "../data/prerank_histogram_30.txt" #&> logs/fold3.log &
#python3 run_model.py schirra_fold4 "../data/5-folds/robust04.qrels_fold_4.train" "../data/qrel_histogram_30.txt" "../data/5-folds/robust04.qrels_fold_4.test" "../data/prerank_histogram_30.txt" #&> logs/fold4.log &
#python3 run_model.py schirra_fold5 "../data/5-folds/robust04.qrels_fold_5.train" "../data/qrel_histogram_30.txt" "../data/5-folds/robust04.qrels_fold_5.test" "../data/prerank_histogram_30.txt" #&> logs/fold5.log &
#
#
#wait

#echo "all processes completed"
