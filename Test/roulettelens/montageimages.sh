#! /bin/sh

echo Running: $0
bindir=`dirname $0`
echo Running from $bindir

. $bindir/config.sh

dir=$2
test $dir || dir=Roulette

baseline=$1
test $dir || baseline=Original

diffdir=$3
test $diffdir || diffdir=diff
montagedir=$4
test $montagedir || montagedir=diff

mkdir -p $montagedir

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
          $CONVERT $baseline/$ff $diffdir/$ff $dir/$ff  +append $montagedir/$ff
          echo $baseline/$ff $diffdir/$ff $dir/$ff  "->" $montagedir/$ff
    done
fi
