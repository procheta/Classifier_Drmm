#$1 classifier output , $2 groundTruthFile
if [ $# -lt 2 ]
then
	echo "classifier output file and ground truth file required"
	exit
fi

file=$1
gtFile=$2
awk '{if ($3 >0.5) {print $1" "$2" 1"}else{print $1" "$2" 0"}}' $file > preprocessed.txt
javac DRMMPreprocessing.java
java DRMMPreprocessing $gtFile preprocessed.txt  result.out 

