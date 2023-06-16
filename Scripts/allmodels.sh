#! /bin/sh

sh build.sh

python3 Python/datagen.py -x 10 -y 50 --model Raytrace --lens SIS --name raytrace -R
python3 Python/datagen.py -x 10 -y 50 --model Roulette --lens SIS --name roulette -R
python3 Python/datagen.py -x 10 -y 50 --model Raytrace --lens SIS --sampled --name raytrace-sampled -R
python3 Python/datagen.py -x 10 -y 50 --model Roulette --lens SIS --sampled --name roulette-sampled -R

python3 Python/datagen.py -x 50 -y 0 --model Raytrace --lens SIS --name raytrace2 -R
python3 Python/datagen.py -x 50 -y 0 --model Roulette --lens SIS --name roulette2 -R
python3 Python/datagen.py -x 50 -y 0 --model Raytrace --lens SIS --sampled --name raytrace-sampled2 -R
python3 Python/datagen.py -x 50 -y 0 --model Roulette --lens SIS --sampled --name roulette-sampled2 -R
