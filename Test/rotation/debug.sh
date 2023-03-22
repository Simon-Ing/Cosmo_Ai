#! /bin/sh

( cd ../../ ; cmake --build build )

pdir=../../Python
dir1=SphereModel-debug
dir2=RouletteSIS-debug
diffdir=diff-debug

mkdir -p $dir1 $dir2 $diffdir

fn=/tmp/spheres.csv

cat > $fn <<EOF 
index,filename,source,chi,x,y,einsteinR,sigma,sigma2,theta,nterms
"s03",image-s03.png,e,50,42,-25,16,8,15,40,12
"s04",image-s04.png,e,50,-38,28,21,5,15,75,12
EOF

python3 $pdir/datagen.py -L s --directory="$dir1" --csvfile $fn --actual --apparent --reflines
python3 $pdir/datagen.py -L sr --directory="$dir2" --csvfile $fn --actual --apparent --reflines
python3 $pdir/compare.py --diff $diffdir $dir1 $dir2

