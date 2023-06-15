#! /bin/sh

python3 Python/datagen.py -x 10 -y 50 --model Raytrace --lens SIS --name raytrace -R
python3 Python/datagen.py -x 10 -y 50 --model Roulette --lens SIS --name roulette -R
python3 Python/datagen.py -x 10 -y 50 --model Raytrace --lens SIS --sampled --name raytrace-sampled -R
python3 Python/datagen.py -x 10 -y 50 --model Roulette --lens SIS --sampled --name roulette-sampled -R
