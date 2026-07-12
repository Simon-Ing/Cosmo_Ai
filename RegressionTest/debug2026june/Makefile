all: sampled.csv unsampled.csv resampled.log reunsampled.log

O=-vvvvv -Z 1024 -z 512

F=dataset.csv

sampled.csv:
	python -m CosmoSim $O --csvfile $F --dir sampled --outfile sampled.csv --sampled --model Roulette | tee sampled.log
unsampled.csv:
	python -m CosmoSim $O --csvfile $F --dir unsampled --outfile unsampled.csv --no-sampled --model Roulette | tee unsampled.log

resampled.log: sampled.csv
	python -m CosmoSim $O --roulette $< --dir resampled | tee resampled.log
reunsampled.log: unsampled.csv
	python -m CosmoSim $O --roulette $< --dir reunsampled | tee reunsampled.log

clean: 
	rm -f *.log
	rm -f sampled.csv unsampled.csv 
