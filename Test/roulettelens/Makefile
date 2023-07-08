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
	datagen.py -Z 600 --csvfile debug.csv --outfile roulette.csv --centred -D Original --actual --original -R | tee images.log 
roulettegen.log: roulette.csv
	mkdir -p Roulette
	roulettegen.py -n 10 -Z 600 --csvfile roulette.csv --centred -D Roulette -R | tee roulettegen.log 
compare.log: roulettegen.log
	compare.py --diff diff/ --masked Original Roulette  | tee compare.log
montage.log:
	./montageimages.sh  | tee montage.log

# Testing
false.log:
	/bin/false | tee false.log
true.log:
	/bin/true | tee true.log
test:
	echo $(CURDIR)
	echo $(PATH)
