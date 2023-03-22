#! /bin/sh

dir=$1
test $dir || dir=Test/`date "+%Y%m%d"`
mkdir -p $dir

baseline=Test/baseline20230320
# baseline=Test/v2.0.3

F=mask centred reflines


for flag in $F plain 
do
   mkdir -p $dir/$flag
   mkdir -p Test/diff/$flag
done

python3 Python/datagen.py --directory="$dir"/plain --csvfile Datasets/debug.csv  || exit 1

for flag in $F
do
  python3 Python/datagen.py --$flag --directory="$dir"/$flag --csvfile Datasets/debug.csv  || exit 2
done

test -d $baseline || exit 3

for flag in $F plain 
do
   python3 Python/compare.py --diff Test/diff/$flag $baseline/$flag $dir/$flag
done
