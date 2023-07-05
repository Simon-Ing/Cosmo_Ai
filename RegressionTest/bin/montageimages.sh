#! /bin/sh

echo Running: $0
bindir=`dirname $0`
echo Running from $bindir

. $bindir/config.sh

dir=$1
test $dir || dir=Test/`date "+%Y%m%d"`
mkdir -p montage-$dir

baseline=$2
test $baseline || baseline=Test/baseline20230704

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
elif test ! -d diff-$dir 
then 
   echo Difference images have not been 
   exit 5 
else
    for f in diff-$dir/*
    do
          ff=`basename $f`
          $CONVERT $baseline/$ff diff-$dir/$ff $dir/$ff  \
                  +append montage-$dir/$flag/$ff
    done
fi
