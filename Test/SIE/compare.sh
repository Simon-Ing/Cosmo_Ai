#! /bin/bash


opt=$*

pdir=../../CosmoSimPy

configlist="srousie sraysie rousie raysie"
CONVERT=convert

shopt -s extglob

for a in diff montage
do
  for b in roulette raytrace compare sampled-compare
  do
     mkdir $a-$b
  done
done

python3 $pdir/compare.py --diff diff-roulette rousie srousie
python3 $pdir/compare.py --diff diff-raytrace raysie sraysie
python3 $pdir/compare.py --diff diff-compare rousie raysie
python3 $pdir/compare.py --diff diff-sampled-compare srousie sraysie



for f in diff-roulette/*
do
  ff=`basename $f`
  $CONVERT \( actual-srousie/actual-$ff rousie/$ff +append \) \( srousie/$ff diff-roulette/$ff +append \) -append montage-roulette/$ff
done

for f in diff-raytrace/*
do
  ff=`basename $f`
  $CONVERT \( actual-sraysie/actual-$ff raysie/$ff +append \) \( sraysie/$ff diff-raytrace/$ff +append \) -append montage-raytrace/$ff
done
for f in diff-compare/*
do
  ff=`basename $f`
  $CONVERT \( actual-rousie/actual-$ff raysie/$ff +append \) \( rousie/$ff diff-compare/$ff +append \) -append montage-compare/$ff
done

for f in diff-sampled-compare/*
do
  ff=`basename $f`
  $CONVERT \( actual-srousie/actual-$ff sraysie/$ff +append \) \( srousie/$ff diff-sampled-compare/$ff +append \) -append montage-sampled-compare/$ff
done
