#! /bin/sh

CONVERT=convert

opt=$*

pdir=../../CosmoSimPy/
fn=dataset.csv

mkdir -p Raytrace actual montage
python3 $pdir/datagen.py $opt --model "Raytrace" --directory="Raytrace" --csvfile $fn  --actual -R --xireference
mv Raytrace/actual*png actual

for m in 3 5 15
do
   mkdir -p Roulette$m  diff$m stat$m
   python3 $pdir/datagen.py $opt --model "Roulette" --directory="Roulette$m" --csvfile $fn  --outfile roulette$m.csv --nterms $m -R --xireference
   python3 $pdir/compare.py -o stat$m -O test.tex --diff diff$m Roulette$m Raytrace
done

for f in Raytrace/*
do
  ff=`basename $f`
  $CONVERT \( actual/actual-$ff Raytrace/$ff +append \) \
     \( Roulette3/$ff diff3/$ff +append \) \
     \( Roulette5/$ff diff5/$ff +append \) \
     \( Roulette15/$ff diff15/$ff +append \) \
     -append montage/$ff
done

texpage ( ) {
   echo '\includegraphics[width=0.33\\textwidth]{'Raytrace/$1.png}
   echo '\includegraphics[width=0.33\\textwidth]{'actual/actual-$1.png}
   echo '\\begin{minipage}[b]{0.33\\textwidth}\centering'
   echo '\\verb|'$1'|'
   echo '\\end{minipage}'

   echo '\includegraphics[width=0.33\\textwidth]{'Roulette3/$1.png}
   echo '\includegraphics[width=0.33\\textwidth]{'diff3/$1.png}
   echo '\\begin{minipage}[b]{0.33\\textwidth}'
   cat stat3/$1.tex
   echo '\\end{minipage}'

   echo '\includegraphics[width=0.33\\textwidth]{'Roulette5/$1.png}
   echo '\includegraphics[width=0.33\\textwidth]{'diff5/$1.png}
   echo '\\begin{minipage}[b]{0.33\\textwidth}'
   cat stat5/$1.tex
   echo '\\end{minipage}'

   echo '\includegraphics[width=0.33\\textwidth]{'Roulette15/$1.png}
   echo '\includegraphics[width=0.33\\textwidth]{'diff15/$1.png}
   echo '\\begin{minipage}[b]{0.33\\textwidth}'
   cat stat15/$1.tex
   echo '\\end{minipage}'

   echo '\clearpage'
}


mkdir -p texpages

cp preamble.tex report.tex

for f in Raytrace/*
do
   ff=`basename $f .png`
   texpage $ff >> report.tex
done

cat tail.tex >> report.tex

