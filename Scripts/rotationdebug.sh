#! /bin/sh

cmake --build build || exit 10

dir=$1
test $dir || dir=Test/current`date "+%Y%m%d"`
mkdir -p $dir

fn=/tmp/spheres.csv

cat > $fn <<EOF
index,filename,source,config,chi,x,y,einsteinR,sigma,theta,sigma2
"ts00",image-ts00.png,t,rs,50,35,0,10,36,0,16
"ts30",image-ts30.png,t,rs,50,35,0,10,36,30,16
"ts60",image-ts60.png,t,rs,50,35,0,10,36,60,16
"ts90",image-ts90.png,t,rs,50,35,0,10,36,90,16
"tp00",image-tp00.png,t,p,50,35,0,10,36,0,16
"tp30",image-tp30.png,t,p,50,35,0,10,36,30,16
"tp60",image-tp60.png,t,p,50,35,0,10,36,60,16
"tp90",image-tp90.png,t,p,50,35,0,10,36,90,16
"ts00",image-tf00.png,t,fs,50,35,0,10,36,0,16
"ts30",image-tf30.png,t,fs,50,35,0,10,36,30,16
"ts60",image-tf60.png,t,fs,50,35,0,10,36,60,16
"ts90",image-tf90.png,t,fs,50,35,0,10,36,90,16
"tp00",image-tr00.png,t,r,50,35,0,10,36,0,16
"tp30",image-tr30.png,t,r,50,35,0,10,36,30,16
"tp60",image-tr60.png,t,r,50,35,0,10,36,60,16
"tp90",image-tr90.png,t,r,50,35,0,10,36,90,16
"es00",image-es00.png,e,rs,50,35,0,10,36,0,16
"es30",image-es30.png,e,rs,50,35,0,10,36,30,16
"es60",image-es60.png,e,rs,50,35,0,10,36,60,16
"es90",image-es90.png,e,rs,50,35,0,10,36,90,16
"ep00",image-ep00.png,e,p,50,35,0,10,36,0,16
"ep30",image-ep30.png,e,p,50,35,0,10,36,30,16
"ep60",image-ep60.png,e,p,50,35,0,10,36,60,16
"ep90",image-ep90.png,e,p,50,35,0,10,36,90,16
"es00",image-ef00.png,e,fs,50,35,0,10,36,0,16
"es30",image-ef30.png,e,fs,50,35,0,10,36,30,16
"es60",image-ef60.png,e,fs,50,35,0,10,36,60,16
"es90",image-ef90.png,e,fs,50,35,0,10,36,90,16
"ep00",image-er00.png,e,r,50,35,0,10,36,0,16
"ep30",image-er30.png,e,r,50,35,0,10,36,30,16
"ep60",image-er60.png,e,r,50,35,0,10,36,60,16
"ep90",image-er90.png,e,r,50,35,0,10,36,90,16
EOF

# python3 CosmoSimPy/datagen.py --imagesize 600 -L pss --directory="$dir" --csvfile $fn --actual --apparent --reflines --psiplot --kappaplot
# python3 CosmoSimPy/datagen.py --imagesize 600 --lens SIS --model Roulette --directory="$dir" --csvfile $fn --actual --apparent --family --reflines --join --maskscale 0.85 --components 8 --showmask
# python3 CosmoSimPy/datagen.py -L sr --directory="$dir" --csvfile $fn --actual --apparent --reflines

python3 CosmoSimPy/datagen.py --csvfile $fn --apparent --actual -R

for i in 00 30 60 90
do
   convert \( actual-image-tp$i.png apparent-image-tp$i.png -append \) \
           \( image-tp$i.png image-ts$i.png -append \) \
           \( image-tr$i.png image-tf$i.png -append \) \
           +append montage$i.png
   convert \( actual-image-ep$i.png apparent-image-ep$i.png -append \) \
           \( image-ep$i.png image-es$i.png -append \) \
           \( image-er$i.png image-ef$i.png -append \) \
           +append montage${i}e.png
done

# Actual   | PM Exact     | Roulette/PM  |
# ---------+--------------+---------------+
# Apparent | Raytrace/SIS | Roulette/SIS   |


echo $dir
