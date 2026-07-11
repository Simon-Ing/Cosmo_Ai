#!/bin/sh

pyd=../../python/

mkdir -p Original Roulette diff montage

python3 -m CosmoSim.datagen -Z 600 --csvfile debug.csv --outfile roulette.csv \
   --model Roulette --no-xireference -D Original --actual  -R --nterms 9 \
   | tee datagen.log || exit 1
python3 -m CosmoSim.roulettegen -n 10 -Z 600 --csvfile roulette.csv \
   --no-xireference -D Roulette -R --nterms 9 \
   | tee roulettegen.log || exit 2
mkdir -p Actual
mv Original/actual-* Actual
python3 $pyd/compare.py --diff diff/ --masked Original Roulette 

for i in diff/*
do
   f=`basename $i`
   convert \( Actual/actual-$f Original/$f +append \) \( diff/$f Roulette/$f +append \) -append montage/$f
done

