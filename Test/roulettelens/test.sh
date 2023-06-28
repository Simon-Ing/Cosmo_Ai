#!/bin/sh

pyd=../../Python

mkdir -p Original Roulette diff montage

python3 $pyd/datagen.py -Z 600 --csvfile debug.csv --outfile roulette.csv --centred -D Original --actual --original | tee datagen.log || exit 1
python3 $pyd/roulettegen.py -n 10 -Z 600 --csvfile roulette.csv --model Roulette --centred -D Roulette --lens Roulette | tee roulettegen.log || exit 2
python3 $pyd/compare.py --diff diff/ Original Roulette

for i in diff/*
do
   f=`basename $i`
   convert \( Original/actual-$f Original/original-$f Original/$f +append \) \( diff/$f Roulette/$f +append \) -append montage/$f
done

