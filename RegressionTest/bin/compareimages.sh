#! /bin/sh

echo Running: $0
bindir=`dirname $0`
echo Running from $bindir

. $bindir/config.sh

dir=$1
test $dir || dir=`date "+%Y%m%d"`
mkdir -p $dir
mkdir -p diff-$dir

baseline=$2
test $baseline || baseline=baseline-test
# 20230704

if test ! -d $baseline 
then 
   echo $baseline does not exist 
   exit 5 
else
   compare.py --diff diff-$dir $baseline $dir --masked
fi
