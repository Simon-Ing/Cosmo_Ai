#!/bin/sh

export PATH=../../Python:$PATH

python3 datagen.py --csvfile debug.csv --outfile roulette.csv --model Raytrace --centred -D Original
python3 roulettegen.py --csvfile roulette.csv --model Roulette --centred -D Roulette
