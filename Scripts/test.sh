#! /bin/sh

dir=$1
test $dir || dir=Test/`date "+%Y%m%d"`
mkdir -p $dir

python3 Python/datagen.py --directory="$dir" --csvfile Datasets/debug.csv  || exit 1
python3 Python/compare.py Test/v2.0.2 $dir Test/diff
