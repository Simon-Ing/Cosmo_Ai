#!/bin/sh

pyd=../../Python

mkdir -p Original Roulette diff

python3 $pyd/datagen.py --csvfile debug.csv --outfile roulette.csv --centred -D Original || exit 1
python3 $pyd/roulettegen.py --csvfile roulette.csv --model Roulette --centred -D Roulette --lens Roulette || exit 2
python3 $pyd/compare.py --diff diff/ Original Roulette
