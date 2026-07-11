all: pm50.txt sis50.txt sie05.txt

sis: sis50.txt sis200.txt
sis100.txt sis50.txt sie05.txt:

P=./Roulettes

%.txt: %.raw
	sort $< > $@

sis%sie.raw:  $P/__main__.py
	time python3 -m CosmoSim.Roulettes.sie --circular --lens SIS --output $@ $* 8
pm%sie.raw:  $P/__main__.py
	time python3 -m CosmoSim.Roulettes.sie --circular --lens PM --output $@ $* 8

sis%.raw:  $P/__main__.py
	time python3 -m CosmoSim.Roulettes --lens SIS --output $@ $* 8
pm%.raw:  $P/__main__.py
	time python3 -m CosmoSim.Roulettes --lens PM --output $@ $* 8

sie%.raw: $P/sie.py 
	time -m CosmoSim.Roulettes.sie --output $@ $* 7
sie%.tex: $P/sie.py 
	time -m CosmoSim.Roulettes.sie --tex $@ $* 7

%.pdf: %.tex
	pdflatex $<

clean:
	rm -f *.log *.raw
cleanall: clean
	rm -f *.txt
