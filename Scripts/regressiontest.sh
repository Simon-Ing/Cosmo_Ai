#! /bin/sh

cmake --build build || exit 10

dir=$1
test $dir || dir=Test/`date "+%Y%m%d"`
mkdir -p $dir

baseline=$2
test $baseline || baseline=Test/baseline20230623
# baseline=Test/v2.0.3

F=mask
# F="mask centred reflines"
# centred and reflines are done in python and are not needed
# when validating C++ code


for flag in $F plain 
do
   mkdir -p $dir/$flag
   mkdir -p Test/diff/$flag
done

python3 Python/datagen.py --directory="$dir"/plain \
   --csvfile Datasets/debug.csv  || exit 1

python3 Python/datagen.py --mask --directory="$dir"/mask \
   --csvfile Datasets/debug.csv  || exit 2

if /bin/false
then
python3 Python/datagen.py --reflines --centred --directory="$dir"/centred \
   --csvfile Datasets/debug.csv  || exit 3
python3 Python/datagen.py --reflines --directory="$dir"/reflines \
   --csvfile Datasets/debug.csv  || exit 4
fi

test -d $baseline || echo $baseline does not exist ; exit 5 

for flag in $F plain 
do
   echo $flag
   python3 Python/compare.py --diff Test/diff/$flag $baseline/$flag $dir/$flag
done

echo $F
