#! /bin/sh

dir=$1
test $dir || dir=`date "+%Y%m%d"`
mkdir -p $dir


python3 ../../Python/datagen.py --directory="$dir" \
   --nterms 5 \
   --outfile "outfile.csv" --csvfile debug.csv  || exit 1

