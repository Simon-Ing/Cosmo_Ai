#!/bin/sh
dir=`date "+%Y%m%d"`
mkdir -p $dir
. ../bin/config.sh
python3 -m CosmoSim -n 10 -Z 600 --roulette roulette.csv -D $dir --no-xireference
