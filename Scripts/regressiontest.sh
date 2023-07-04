#! /bin/sh

if test x$COSMOSIM_DEVELOPING = xyes
then
   cmake --build build || exit 10
fi

dir=$1
test $dir || dir=Test/`date "+%Y%m%d"`
mkdir -p $dir

baseline=$2
test $baseline || baseline=Test/baseline20230703

F=mask
# F="mask centred reflines"
# centred and reflines are done in python and are not needed
# when validating C++ code


for flag in $F plain 
do
   mkdir -p $dir/$flag
   mkdir -p Test/diff/$flag
   mkdir -p Test/montage/$flag
done

python3 CosmoSimPy/datagen.py --directory="$dir"/plain \
   --csvfile Datasets/debug.csv  || exit 1

python3 CosmoSimPy/datagen.py --mask --directory="$dir"/mask \
   --csvfile Datasets/debug.csv  || exit 2

if /bin/false
then
python3 CosmoSimPy/datagen.py --reflines --centred --directory="$dir"/centred \
   --csvfile Datasets/debug.csv  || exit 3
python3 CosmoSimPy/datagen.py --reflines --directory="$dir"/reflines \
   --csvfile Datasets/debug.csv  || exit 4
fi

if test -d $baseline 
then 
  if test -x "`which convert`"
  then

    for flag in $F plain 
    do
       echo $flag
       python3 CosmoSimPy/compare.py --diff Test/diff/$flag \
          $baseline/$flag $dir/$flag --masked
       for f in Test/diff/$flag/*
       do
          ff=`basename $f`
          echo "$ff" - "$f"
          echo "$baseline/$flag/$ff Test/diff/$flag/$ff $dir/$flag/$ff +append Test/montage/$flag/$ff"
          if test x$OSTYPE == $xcygwin || test x$OSTYPE == $xmsys
          then
              convert $baseline\$flag\$ff Test\diff\$flag\$ff $dir\$flag\$ff +append Test\montage\$flag\$ff
          else
              convert $baseline/$flag/$ff Test/diff/$flag/$ff $dir/$flag/$ff  \
                  +append Test/montage/$flag/$ff
          fi

       done
    done

    echo $F
  else
     echo basename test: `basename /foo/bar/test/image.png` \
          "(basename /foo/bar/test/image.png)"
     echo ImageMagick is not installed 
     exit 6 
  fi
else
   echo $baseline does not exist 
   exit 5 
fi
