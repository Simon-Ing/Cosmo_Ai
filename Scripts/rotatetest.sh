#! /bin/sh

dir=$1
test $dir || dir=Test/rotate`date "+%Y%m%d"`
mkdir -p $dir

fn=/tmp/ts.csv 

cat > $fn <<EOF
index,filename,source,lens,chi,x,y,einsteinR,sigma,sigma2,theta,nterms
"sr01",image-s01.png,s,sr,50,75,75,100,20,0,0,32
"s01",image-s01.png,s,s,50,75,75,100,20,0,0,32
"sr02",image-s02.png,s,sr,50,40,20,30,15,0,0,32
"s02",image-s02.png,s,s,50,40,20,30,15,0,0,32
"sr03",image-s03.png,e,sr,50,40,20,30,15,30,40,32
"s03",image-s03.png,e,s,50,40,20,30,15,30,40,32
EOF

python3 CosmoSimPy/datagen.py --directory="$dir" --csvfile $fn --actual --apparent --reflines


# "ss15",image-ss15.png,t,ss,50,10,0,7,20,0,0,16
# "ss16",image-ss16.png,t,ss,50,50,0,7,20,0,0,16
# "ss17",image-ss16.png,t,ss,50,50,0,20,20,0,0,16
# "s15",image-s15.png,t,s,50,10,0,7,20,0,0,16
# "s16",image-s16.png,t,s,50,50,0,7,20,0,0,16
# "s17",image-s16.png,t,s,50,50,0,20,20,0,0,16
