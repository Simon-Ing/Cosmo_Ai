
esrc=-S e -2 20 -t 45
sis=-L -n 16

bin/makeimage -N pm-sphere -R -I 1024 -x 20 -X 50 -y 0 -E 10 -s 10
bin/makeimage -N pm-e -R -I 1024 -x 20 -X 50 -y 0 -E 10 -s 10 $esrc
bin/makeimage -N sis-e -R -I 1024 -x 20 -X 50 -y 0 -E 10 -s 10 $esrc $sis
bin/makeimage -N sis-sphere -R -I 1024 -x 20 -X 50 -y 0 -E 10 -s 10  $sis
