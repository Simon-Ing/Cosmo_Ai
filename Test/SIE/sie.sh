#! /bin/sh

( cd ../.. && cmake --build build ) || exit 1

opt=$*

pdir=../../CosmoSimPy/
fn=../sie.csv
# Note, origin.csv specifies scenarioes with the almost source at the origin.

configlist="srousie sraysie rousie raysie"
# configlist='"Sampled Roulette SIS" "Sampled Raytrace SIS" "Roulette SIS" "Raytrace SIS"'


for dir in $configlist
do
   echo "Testing" $dir
   mkdir -p $dir actual-$dir
   python3 $pdir/datagen.py $opt --config "$dir" --directory="$dir" --csvfile $fn  --actual -R -n 5
   mv $dir/actual*png actual-$dir
done


