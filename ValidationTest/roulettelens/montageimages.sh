#! /bin/sh

echo Running: $0
bindir=`dirname $0`
echo Running from $bindir

. $bindir/config.sh

dir=$2
test $dir || dir=roulette-images

baseline=$1
test $baseline || baseline=Original

diffdir=$3
test $diffdir || diffdir=diff

montagedir=$4
test $montagedir || montagedir=montage

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
   echo Images have not been created: $dir
   exit 5 
elif test ! -d $diffdir
then 
   echo Difference images have not been created: $diffdir
   exit 5 
else
    for f in $diffdir/*
    do
          ff=`basename $f`
          echo $CONVERT $baseline/$ff $diffdir/$ff $dir/$ff  "->" $montagedir/$ff
          $CONVERT $baseline/$ff $diffdir/$ff $dir/$ff  +append $montagedir/$ff
    done
fi
