#!/bin/sh
dir=`date "+%Y%m%d"`
mkdir -p $dir
roulettegen.py -n 10 -Z 600 --csvfile roulette.csv -D $dir 
