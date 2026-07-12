#! /bin/sh
# Deprecated in favour of RegressionTest/cluster

. ../../pythonenv/bin/activate

( cd ../.. && cmake --build build ) || exit 1

opt=$*

pdir=../../CosmoSimPy/

mkdir -p cluster

python3 $pdir/datagen.py $opt --model Raytrace --csvfile ../../RegressionTest/cluster.csv --directory=cluster --actual -R 
