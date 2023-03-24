#! /bin/sh

cmake --build build

dir=$1
test $dir || dir=Test/current`date "+%Y%m%d"`
mkdir -p $dir

fn=/tmp/spheres.csv

cat > $fn <<EOF
index,filename,source,chi,x,y,einsteinR,sigma,sigma2,theta,nterms
"s01",image-s01.png,s,50,50,50,40,10,0,0,50
"s02",image-s02.png,s,50,35,24,20,12,0,0,50
"s03",image-s03.png,e,50,42,-25,16,8,15,40,50
"s04",image-s04.png,e,50,-38,28,21,5,15,75,50
"s07",image-s07.png,s,50,-20,-20,10,15,0,0,50
"s08",image-s08.png,s,50,30,0,12,15,0,0,50
"s11",image-s01.png,s,50,30,30,40,20,0,0,50
"s12",image-s02.png,s,50,35,24,30,30,0,0,50
"s13",image-s02.png,s,50,40,40,60,40,0,0,50
"s14",image-s02.png,e,50,40,40,60,60,6,135,50
EOF

python3 Python/datagen.py --imagesize 600 -L sr --directory="$dir" --csvfile $fn --actual --apparent --family --reflines
# python3 Python/datagen.py -L sr --directory="$dir" --csvfile $fn --actual --apparent --reflines


# "ss15",image-ss15.png,t,ss,50,10,0,7,20,0,0,16
# "ss16",image-ss16.png,t,ss,50,50,0,7,20,0,0,16
# "ss17",image-ss16.png,t,ss,50,50,0,20,20,0,0,16
# "s15",image-s15.png,t,s,50,10,0,7,20,0,0,16
# "s16",image-s16.png,t,s,50,50,0,7,20,0,0,16
# "s17",image-s16.png,t,s,50,50,0,20,20,0,0,16

