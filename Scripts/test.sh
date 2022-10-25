#!/bin.sh

esrc="-S e -2 20"
esrc1="$esrc -t 45"
sis="-L -n 16"
opt="-R -I 512 -x 20 -X 50 -y 0 -E 10 -s 10"
makeimage -N pm-sphere $opt
makeimage -N sis-sphere $opt $sis

makeimage -N pm-e00- -t 0  $opt $esrc
makeimage -N pm-e30- -t 30  $opt $esrc
makeimage -N pm-e60- -t 60  $opt $esrc
makeimage -N pm-e90- -t 90  $opt $esrc

makeimage -N sis-e00- -t 0 $opt $esrc $sis
makeimage -N sis-e30- -t 30 $opt $esrc $sis
makeimage -N sis-e60- -t 60 $opt $esrc $sis
makeimage -N sis-e90- -t 90 $opt $esrc $sis
