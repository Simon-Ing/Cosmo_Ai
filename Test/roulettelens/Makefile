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

alltests:  montage.log montage-eta.log montage-eta-orig.log

test.log: roulette.csv
	mkdir -p Original test{1,2}images
	datagen.py -Z 600 --csvfile debug.csv --outfile test1.csv --centred -D test1images
	datagen.py -Z 600 --csvfile debug.csv --outfile test2.csv -D test2images
	touch $@
roulette.csv: debug.csv
	mkdir -p Original
	datagen.py -Z 600 --csvfile $< --outfile $@ --centred -D Original --actual --original | tee images.log 
%-gen.log: %.csv
	mkdir -p $*-images
	roulettegen.py -n 10 -Z 600 --csvfile $< -D $*-images
%-reflines.log: %.csv
	mkdir -p $*-images
	roulettegen.py -n 10 -Z 600 --csvfile $< -D $*-images -R
%-compare.log: %-gen.log
	mkdir -p diff
	compare.py --diff diff --masked Original $*-images | tee $@ ; exit 0
compare-eta.log: roulette-gen.log noeta-gen.log
	mkdir -p eta-diff
	compare.py --diff eta-diff/ --masked roulette-images noeta-images | tee $@ ; exit 0
compare-eta-orig.log: roulette-gen.log noeta-gen.log
	mkdir -p eta-orig-diff
	compare.py --diff eta-orig-diff/ --masked Original noeta-images | tee $@ ; exit 0
montage.log: roulette-compare.log
	./montageimages.sh  | tee montage.log
montage-eta.log: compare-eta.log
	./montageimages.sh  roulette-images noeta-images eta-diff  eta-montage | tee montage.log
montage-eta-orig.log: noeta-gen.log compare-eta-orig.log 
	./montageimages.sh  Original noeta-images eta-orig-diff  eta-orig-montage | tee montage.log

# Testing
false.log:
	/bin/false | tee false.log
true.log:
	/bin/true | tee true.log
test:
	echo $(CURDIR)
	echo $(PATH)
