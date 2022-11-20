#!/bin/sh

# This script reads a CSV file with source and lens parameters and 
# generate corresponding images.
#
# Useful options:
# -R    to draw axes and box the plot
# -Z n  to change the image size

pwd
echo $0 $1
which makeimage

IFS=,
read -r header
echo $header
while read -r idx fn srcmode lensmode chi R phi einsteinr sigma sigma2 theta nterms x y
do
  idx=`echo $idx | tr -d '"'`
  makeimage $* -X $chi -x $R -T $phi -E $einsteinr \
	        -s $sigma -2 $sigma2 -t $theta -n $nterms \
                -S $srcmode -L $lensmode -N $idx 
  centreimage.py $fn
done
