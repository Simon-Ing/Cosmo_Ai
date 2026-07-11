#! /bin/sh


pyd=../../CosmoSimPy

mkdir Raytrace Original
python3 $pyd/datagen.py -Z 600 --csvfile dataset.csv \
   --model Raytrace --lens SIS --xireference -D Raytrace  -R 
python3 $pyd/datagen.py -Z 600 --csvfile dataset.csv --outfile roulette.csv \
   --model Roulette --lens SIS --xireference -D Original --actual  -R --nterms 5

./tweak.py
./txt.py

for i in roulette-*.csv
do
   f=`basename $i .csv`
   mkdir $f
   echo Generating images for $f
   python3 $pyd/roulettegen.py -Z 600 --csvfile $i --xireference -D $f -R --nterms 5
done

pdflatex tweak
