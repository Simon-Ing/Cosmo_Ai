#! /bin/sh

cmake --build build

python3 Python/datagen.py -x 10 -y 50 --model Raytrace --lens SIS --name raytrace -R --actual
python3 Python/datagen.py -x 10 -y 50 --model Roulette --lens SIS --name roulette -R --actual
python3 Python/datagen.py -x 10 -y 50 --model Raytrace --lens SIS --sampled --name raytrace-sampled -R --actual
python3 Python/datagen.py -x 10 -y 50 --model Roulette --lens SIS --sampled --name roulette-sampled -R --actual

python3 Python/datagen.py -x 50 -y 10 --model Raytrace --lens SIS --name raytrace2 -R --actual
python3 Python/datagen.py -x 50 -y 10 --model Roulette --lens SIS --name roulette2 -R --actual
python3 Python/datagen.py -x 50 -y 10 --model Raytrace --lens SIS --sampled --name raytrace-sampled2 -R --actual
python3 Python/datagen.py -x 50 -y 10 --model Roulette --lens SIS --sampled --name roulette-sampled2 -R --actual

python3 Python/datagen.py -x 35 --y=-29 --einsteinradius 3 --sigma 43 --model Raytrace --lens SIS --name raytrace3 -R --actual
python3 Python/datagen.py -x 35 --y=-29 --einsteinradius 3 --sigma 43 --model Raytrace --lens SIS --sampled --name raytrace-sampled3 -R --actual

python3 Python/datagen.py -x=-50 -y=-10 --model Raytrace --lens SIS --name raytrace4 -R --actual
python3 Python/datagen.py -x=-50 -y=-10 --model Roulette --lens SIS --name roulette4 -R --actual
python3 Python/datagen.py -x=-50 -y=-10 --model Raytrace --lens SIS --sampled --name raytrace-sampled4 -R --actual
python3 Python/datagen.py -x=-50 -y=-10 --model Roulette --lens SIS --sampled --name roulette-sampled4 -R --actual
