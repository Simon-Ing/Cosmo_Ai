#! /bin/sh

echo Running: $0
bindir=`dirname $0`
echo Running from $bindir

. $bindir/config.sh

dir=$1
test $dir || dir=`date "+%Y%m%d"`
mkdir -p montage

baseline=$2
test $baseline || baseline=baseline20230705
#20230704

if test -z "$CONVERT"
then
     echo ImageMagick is not installed 
     exit 6 
elif test ! -d $baseline 
then 
   echo $baseline does not exist 
   exit 5 
elif test ! -d $dir 
then 
   echo Images have not been 
   exit 5 
elif test ! -d diff
then 
   echo Difference images have not been 
   exit 5 
else
    for f in diff/*
    do
          ff=`basename $f`
          $CONVERT $baseline/$ff diff/$ff $dir/$ff  +append montage/$ff
    done
fi
