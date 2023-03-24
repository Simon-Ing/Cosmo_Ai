#! /bin/sh

cmake --build build

dir=$1
test $dir || dir=Test/current`date "+%Y%m%d"`
mkdir -p $dir

fn=Test/spheres.csv

python3 Python/datagen.py -L sr --directory="$dir" --csvfile $fn --actual --apparent --family --reflines
# python3 Python/datagen.py -L sr --directory="$dir" --csvfile $fn --actual --apparent --reflines


# "ss15",image-ss15.png,t,ss,50,10,0,7,20,0,0,16
# "ss16",image-ss16.png,t,ss,50,50,0,7,20,0,0,16
# "ss17",image-ss16.png,t,ss,50,50,0,20,20,0,0,16
# "s15",image-s15.png,t,s,50,10,0,7,20,0,0,16
# "s16",image-s16.png,t,s,50,50,0,7,20,0,0,16
# "s17",image-s16.png,t,s,50,50,0,20,20,0,0,16

