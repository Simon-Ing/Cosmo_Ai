#! /bin/sh

. ../../pythonenv/bin/activate

( cd ../.. && cmake --build build && cmake --install build --prefix . ) || exit 1


opt=$*

pdir=../../CosmoSimPy/
fn=../sie.csv
# Note, origin.csv specifies scenarioes with the almost source at the origin.

configlist="srousie sraysie rousie raysie"
# configlist='"Sampled Roulette SIS" "Sampled Raytrace SIS" "Roulette SIS" "Raytrace SIS"'

mkdir -p singleton reference diff actual montage

python3 $pdir/datagen.py $opt --model Raytrace --csvfile singleton.csv --directory=singleton --actual -R 
python3 $pdir/datagen.py $opt --model Raytrace --csvfile reference.csv --directory=reference -R 
python3 $pdir/compare.py --diff diff singleton reference

mkdir -p singleton-roulette reference-roulette diff-roulette montage-roulette
python3 $pdir/datagen.py $opt --model Roulette --csvfile singleton.csv --directory=singleton-roulette --actual -R  -n 5
python3 $pdir/datagen.py $opt --model Roulette --csvfile reference.csv --directory=reference-roulette -R -n 5
python3 $pdir/compare.py --diff diff-roulette singleton-roulette reference-roulette

mv singleton/actual* actual

for f in diff/*
do
  ff=`basename $f`
  convert \( actual/actual-$ff singleton/$ff +append \) \( reference/$ff diff/$ff +append \) -append montage/$ff
  convert \( actual/actual-$ff singleton-roulette/$ff +append \) \( reference-roulette/$ff diff-roulette/$ff +append \) -append montage-roulette/$ff
done

