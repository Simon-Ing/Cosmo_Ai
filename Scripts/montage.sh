#!/bin/sh

for i 
do
   idx=`basename $i .png | cut -d- -f2`
   dir=`dirname $i`
   convert $dir/apparent-$idx.png $dir/actual-$idx.png $dir/image-$idx.png +append $dir/montage$idx.png
done
