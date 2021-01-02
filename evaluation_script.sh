#A light-weight AWK script to compute accuracy/precision/recall (macro-averaged) given an input file of space separated
#predicted and reference values.
#Could easily be extended to multiple classes.

if [ $# -lt 1 ]
then
    echo "Usage: $0 <pred-rel> file" 
    exit
fi

RESFILE=$1 
cat $RESFILE | awk '{if ($1==$2) c++;} END {print "Accuracy = " c/NR}'

#Recall
s=""
for LABEL in 0 1
do
    recall=`cat $RESFILE | awk -v label="$LABEL" '{if ($2==label) print $0}' | awk -v label="$LABEL" '{if ($1==$2) c++;} END {printf("Recall(class: %s) = %.4f\n", label, c/NR)}'`
    echo $recall
    recall=`echo $recall| awk '{print $NF}'`
    s=$s" "$recall
done
echo $s | awk '{printf("Recall = %.4f\n",( $1 + $2)/2)}'

#Precision
s=""
for LABEL in 0 1
do
    prec=`cat $RESFILE | awk -v label="$LABEL" '{if ($1==label) print $0}' | awk -v label="$LABEL" '{if ($1==$2) c++;} END {printf("Precision(class: %s) = %.4f\n", label, c/NR)}'`
    echo $prec
    prec=`echo $prec| awk '{print $NF}'`
    s=$s" "$prec
done
echo $s | awk '{printf("Precision = %.4f\n",( $1 + $2)/2)}'