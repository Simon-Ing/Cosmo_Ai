#!/bin/sh

# This script reads a CSV file with source and lens parameters and 
# generate corresponding images.

IFS=,
read -r header
echo $header
while read -r idx fn srcmode lensmode chi x y einsteinr sigma sigma2 theta nterms
do
  idx=`echo $idx | tr -d '"'`
  bin/makeimage $* -X $chi -x $x -y $y -E $einsteinr -s $sigma -2 $sigma2 -t $theta -n $nterms \
                -S $srcmode -L $lensmode -N $idx -Z 800
  convert apparent-$idx.png actual-$idx.png image-$idx.png secondary-$idx.png +append montage$idx.png
done
