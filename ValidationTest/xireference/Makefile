# (C) 2023: Hans Georg Schaathun <georg@schaathun.net> 

pyd=../../CosmoSimPy
export PATH := $(pyd):../../RegressionTest/bin:$(PATH)
SHELL=/bin/bash -o pipefail

CONVERT=convert


all: alltests

status: 
	cat */*.status
clean:
	rm -f *.status *.log
	rm -f roulette.csv
	rm -rf diff *-diff montage *-montage *-images Original

alltests:  montage.log 

roulette.csv: debug.csv
	mkdir -p Original
	datagen.py --xireference -Z 600 --nterms 10 --csvfile $< --outfile $@ --centred -D Original | tee images.log 
%-gen.log: %.csv
	mkdir -p $*-images
	roulettegen.py  --xireference -n 10 -Z 600 --csvfile $< -D $*-images
%-reflines.log: %.csv
	mkdir -p $*-images
	roulettegen.py  --xireference -n 10 -Z 600 --csvfile $< -D $*-images -R
%-compare.log: %-gen.log
	mkdir -p diff
	compare.py --diff diff --masked Original $*-images | tee $@ ; exit 0
montage.log: roulette-compare.log 
	./montageimages.sh  | tee montage.log

# Testing
false.log:
	/bin/false | tee false.log
true.log:
	/bin/true | tee true.log
test:
	echo $(CURDIR)
	echo $(PATH)
