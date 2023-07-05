#! /bin/sh

echo Running: $0
bindir=`dirname $0`
echo Running from $bindir

. $bindir/config.sh

dir=$1
test $dir || dir=`date "+%Y%m%d"`
mkdir -p $dir

datagen.py --directory="$dir" --csvfile debug.csv  || exit 1

