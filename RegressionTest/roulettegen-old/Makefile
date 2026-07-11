# (C) 2023: Hans Georg Schaathun <georg@schaathun.net> 

include ../makefile-template

roulette.csv: orig.csv
	mkdir -p original
	python -m CosmoSimPy --directory=original --outfile $@ --csvfile $<

