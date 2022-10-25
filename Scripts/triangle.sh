#!/bin/sh

tsrc="-S t"
sis="-L -n 16"
opt="-R -I 512 -x 20 -X 50 -y 0 -E 10 -s 24"

bin/makeimage -N pm-t00- -t 0  $opt $tsrc
bin/makeimage -N pm-t30- -t 30  $opt $tsrc
bin/makeimage -N pm-t60- -t 60  $opt $tsrc
bin/makeimage -N pm-t90- -t 90  $opt $tsrc

bin/makeimage -N sis-t00- -t 0 $opt $tsrc $sis
bin/makeimage -N sis-t30- -t 30 $opt $tsrc $sis
bin/makeimage -N sis-t60- -t 60 $opt $tsrc $sis
bin/makeimage -N sis-t90- -t 90 $opt $tsrc $sis
