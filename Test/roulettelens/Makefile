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

alltests:  montage.log

roulette.csv:
	mkdir -p Original
	datagen.py -Z 600 --csvfile debug.csv --outfile $@ --centred -D Original --actual --original -R | tee images.log 
%-gen.log: %.csv
	mkdir -p Roulette
	roulettegen.py -n 10 -Z 600 --csvfile $< -D $*-images
%-compare.log: %-gen.log
	compare.py --diff diff/ --masked Original $*-images | tee $@ ; exit 0
compare-eta.log: roulette-gen.log noeta-gen.log
	compare.py --diff diff/ --masked roulette-images noeta-images | tee $@ ; exit 0
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
