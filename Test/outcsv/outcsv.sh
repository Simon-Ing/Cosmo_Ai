#! /bin/sh

cmake --build build || exit 10

dir=$1
test $dir || dir=Test/outcsv`date "+%Y%m%d"`
mkdir -p $dir


python3 Python/datagen.py --directory="$dir"/plain \
   --outfile "$dir/outfile.csv" --csvfile debug.csv  || exit 1

