#! /bin/sh

pdir=../../Python
dir1=Old
dir2=New
diffdir=diff

mkdir -p $dir1 $dir2 $diffdir

fn=../spheres.csv

python3 $pdir/datagen.py -L sr --directory="$dir1" --csvfile $fn 
python3 $pdir/datagen.py -L rs --directory="$dir2" --csvfile $fn 
python3 $pdir/compare.py --diff $diffdir $dir1 $dir2

