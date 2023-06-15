#! /bin/sh

python3 Python/datagen.py -x 10 -y 50 --model Raytrace --lens SIS --name raytrace
python3 Python/datagen.py -x 10 -y 50 --model Roulette --lens SIS --name roulette
python3 Python/datagen.py -x 10 -y 50 --model Raytrace --lens SIS --sampled --name raytrace-sampled
python3 Python/datagen.py -x 10 -y 50 --model Roulette --lens SIS --sampled --name roulette-sampled
