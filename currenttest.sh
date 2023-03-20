#! /bin/sh

dir=$1
test $dir || dir=Test/sampled`date "+%Y%m%d"`
mkdir -p $dir

fn=/tmp/ts.csv 

cat > $fn <<EOF
index,filename,source,lens,chi,x,y,einsteinR,sigma,sigma2,theta,nterms
"sampled01",image-sampled01.png,s,ss,50,75,75,100,20,0,0,32
"s01",image-s01.png,s,s,50,75,75,100,20,0,0,32
"sampled05",image-sampled05.png,s,ss,50,10,20,7,20,0,0,32
"sampled06",image-sampled06.png,s,ss,50,50,25,7,20,0,0,32
"s05",image-s05.png,s,s,50,10,10,7,20,0,0,32
"s06",image-s06.png,s,s,50,50,75,7,20,0,0,32
EOF

python3 Python/datagen.py --directory="$dir" --csvfile $fn --actual --apparent --reflines


# "ss15",image-ss15.png,t,ss,50,10,0,7,20,0,0,16
# "ss16",image-ss16.png,t,ss,50,50,0,7,20,0,0,16
# "ss17",image-ss16.png,t,ss,50,50,0,20,20,0,0,16
# "s15",image-s15.png,t,s,50,10,0,7,20,0,0,16
# "s16",image-s16.png,t,s,50,50,0,7,20,0,0,16
# "s17",image-s16.png,t,s,50,50,0,20,20,0,0,16
