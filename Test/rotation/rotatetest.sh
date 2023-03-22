#! /bin/sh

pdir=../../Python
dir1=SphereModel
dir2=RouletteSIS
diffdir=diff

mkdir $dir1 $dir2 $diffdir

fn=/tmp/ts.csv 

cat > $fn <<EOF
index,filename,source,chi,x,y,einsteinR,sigma,sigma2,theta,nterms
"s01",image-s01.png,s,50,75,75,100,20,0,0,12
"s02",image-s02.png,s,50,40,20,30,15,0,0,12
"s03",image-s03.png,e,50,40,20,30,15,30,40,12
"s04",image-s03.png,e,-20,40,10,30,10,30,115,12
"s05",image-s01.png,s,50,75,75,100,20,0,0,28
"s06",image-s02.png,s,50,40,20,30,15,0,0,28
EOF

python3 $pdir/datagen.py -L s --directory="$dir1" --csvfile $fn 
python3 $pdir/datagen.py -L sr --directory="$dir2" --csvfile $fn 

