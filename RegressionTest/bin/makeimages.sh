#! /bin/sh

echo Running: $0
bindir=`dirname $0`
echo Running from $bindir

# Default
opts="-Z 400"

if test -f ./makeimages.sh
then
   echo Starting ./makeimages.sh
   exec sh ./makeimages.sh
fi

. $bindir/config.sh

dir=$1
test $dir || dir=`date "+%Y%m%d"`
mkdir -p $dir

test $csv || csv=debug.csv
echo $csv

python -m CosmoSim $opts --directory="$dir" --csvfile $csv  || exit 1
