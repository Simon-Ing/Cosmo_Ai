#! /bin/sh

. ../../pythonenv/bin/activate

( cd ../.. && cmake --build build && cmake --install build --prefix . ) || exit 1

opt=$*

pdir=../../CosmoSimPy/

# Note, origin.csv specifies scenarioes with the almost source at the origin.

mkdir -p singleton-roulette reference-roulette diff-roulette montage-roulette

python3 $pdir/datagen.py $opt --model Roulette --csvfile singleton1.csv --directory=singleton-roulette --actual -R  -n 5 >roulette-sin.log 
python3 $pdir/datagen.py $opt --model Roulette --csvfile reference1.csv --directory=reference-roulette -R -n 5 >roulette-ref.log 
python3 $pdir/compare.py --diff diff-roulette reference-roulette singleton-roulette 

mkdir -p actual
mv singleton-roulette/actual* actual

for f in diff-roulette/*
do
  ff=`basename $f`
  convert \( actual/actual-$ff singleton-roulette/$ff +append \) \( reference-roulette/$ff diff-roulette/$ff +append \) -append montage-roulette/$ff
done

