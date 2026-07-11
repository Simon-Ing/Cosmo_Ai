#! /bin/bash


opt=$*

pdir=../../CosmoSimPy

configlist="ss pss fs rs"
CONVERT=convert

shopt -s extglob

for a in diff montage
do
  for b in roulette raytrace compare sampled-compare
  do
     mkdir $a-$b
  done
done

python3 $pdir/compare.py --diff diff-roulette fs ss
python3 $pdir/compare.py --diff diff-raytrace rs pss
python3 $pdir/compare.py --diff diff-compare fs rs
python3 $pdir/compare.py --diff diff-sampled-compare ss pss



for f in diff-roulette/*
do
  ff=`basename $f`
  $CONVERT \( actual-ss/actual-$ff fs/$ff +append \) \( ss/$ff diff-roulette/$ff +append \) -append montage-roulette/$ff
done

for f in diff-raytrace/*
do
  ff=`basename $f`
  $CONVERT \( actual-pss/actual-$ff rs/$ff +append \) \( pss/$ff diff-raytrace/$ff +append \) -append montage-raytrace/$ff
done
for f in diff-compare/*
do
  ff=`basename $f`
  $CONVERT \( actual-fs/actual-$ff rs/$ff +append \) \( fs/$ff diff-compare/$ff +append \) -append montage-compare/$ff
done

for f in diff-sampled-compare/*
do
  ff=`basename $f`
  $CONVERT \( actual-ss/actual-$ff pss/$ff +append \) \( ss/$ff diff-sampled-compare/$ff +append \) -append montage-sampled-compare/$ff
done
