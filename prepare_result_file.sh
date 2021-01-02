file=$1
gtFile=$2
awk '{if ($3 >0.5) {print $1" "$2" 1"}else{print $1" "$2" 0"}}' $file > preprocessed.txt
javac DRMMPreprocessing.java
java DRMMPreprocessing $2 $gtFile result.out 

