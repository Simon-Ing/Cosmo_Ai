# (C) 2023: Hans Georg Schaathun <georg@schaathun.net> 

%.baseline: ../../CosmoSimPy/CosmoSim/%.txt
	sort $< > $@
%.new: %.txt
	sort $< > $@
%.txt:
	python3 ../../CosmoSimPy/amplitudes.py --lens SIS $* 16
%.status: %.new %.baseline
	diff $^ && echo Regression test passes > $@

