# (C) 2023: Hans Georg Schaathun <georg@schaathun.net> 

pyd=../../CosmoSimPy
export PATH := $(pyd):../../RegressionTest/bin:$(PATH)
SHELL=/bin/bash -o pipefail

CONVERT=convert


all: alltests

clean:
	rm -f *.status *.log
	rm -f roulette.csv
	rm -rf diff *-diff montage *-montage *-images Original

alltests:  roulette-montage.log 

roulette.csv: debug.csv
	mkdir -p Original
	datagen.py -Z 600 --centred --model Raytrace --nterms 10 --csvfile $< --outfile $@ -D Original | tee images.log 
roulette.log: debug.csv
	mkdir -p Roulette
	datagen.py -Z 600 --centred --model Roulette --nterms 10 --csvfile $< --outfile $@ -D Roulette | tee images.log 
%-gen.log: %.csv
	mkdir -p $*-images
	roulettegen.py -n 10 -Z 600 --csvfile $< -D $*-images | tee $@
%-reflines.log: %.csv
	mkdir -p $*-images
	roulettegen.py -n 10 -Z 600 --csvfile $< -D $*-images -R | tee $@
%-compare.log: %-gen.log
	mkdir -p diff
	compare.py --diff diff --masked Original $*-images | tee $@ ; exit 0
compare.log: roulette.csv roulette.log
	mkdir -p diff2
	compare.py --diff diff2 --masked Original Roulette | tee $@ ; exit 0
roulette-montage.log: roulette-compare.log 
	./montageimages.sh  | tee $@
montage.log: roulette.csv compare.log 
	./montageimages.sh Original Roulette diff2 montage2 | tee $@

# Testing
false.log:
	/bin/false | tee false.log
true.log:
	/bin/true | tee true.log
test:
	echo $(CURDIR)
	echo $(PATH)
