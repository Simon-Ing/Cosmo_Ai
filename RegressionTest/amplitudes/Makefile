# (C) 2023: Hans Georg Schaathun <georg@schaathun.net> 

all: sie05.status sis50.status pm50.status

sie05.new sis50.new:
sie05.baseline:
sis50.baseline:

P=../../python

%.baseline: ../../src/CosmoSim/%.txt
	sort $< > $@
%.new: %.txt
	sort $< > $@
pm%.txt:
	echo $*
	python3 -m CosmoSim.Roulettes --output $@ --lens PM $* 
sis%.txt:
	echo $*
	python3 -m CosmoSim.Roulettes --output $@ --lens SIS $* 
sie%.txt:
	python3 -m CosmoSim.Roulettes.sie --output $@ --lens SIE $* 
%.status: %.new %.baseline
	( diff $^ && echo Regression test succeeds || echo Amplitudes discrepancy ) > $@

clean:
	rm -f 50.* *.txt *.new
	rm -f *.status

alltests: all
