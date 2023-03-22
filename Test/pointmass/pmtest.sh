#! /bin/sh

( cd ../.. && cmake --build build )

pdir=../../Python
dir1=Exact
dir2=Roulette
diffdir=diff

mkdir -p $dir1 $dir2 $diffdir

fn=../sphere.csv

python3 $pdir/datagen.py -L p --directory="$dir1" --csvfile $fn 
python3 $pdir/datagen.py -L r --directory="$dir2" --csvfile $fn 
python3 $pdir/compare.py --diff $diffdir $dir1 $dir2

