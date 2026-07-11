#! /bin/sh

( cd ../.. && cmake --build build ) || exit 1

opt=$*

pdir=../../CosmoSimPy/
fn=../sis.csv

configlist="ss pss fs rs"
# configlist='"Sampled Roulette SIS" "Sampled Raytrace SIS" "Roulette SIS" "Raytrace SIS"'


for dir in $configlist
do
   echo "Testing" $dir
   mkdir -p $dir actual-$dir
   python3 -m CosmoSim.datagen $opt --config "$dir" --directory="$dir" --csvfile $fn  --actual -R || exit 2
   mv $dir/actual*png actual-$dir
done


