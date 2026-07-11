#! /bin/sh
# Deprecated in favour of RegressionTest/cluster

. ../../pythonenv/bin/activate

( cd ../.. && cmake --build build ) || exit 1

opt=$*

pdir=../../CosmoSimPy/

mkdir -p constellation

python3 $pdir/datagen.py $opt --model Raytrace --csvfile constellation.csv --directory=constellation --actual -R 
