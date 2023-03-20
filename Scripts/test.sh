#! /bin/sh

dir=$1
test $dir || dir=Test/`date "+%Y%m%d"`
mkdir -p $dir

baseline=baseline20230320

python3 Python/datagen.py --directory="$dir" --csvfile Datasets/debug.csv  || exit 1
python3 Python/compare.py --diff Test/diff $baseline $dir
