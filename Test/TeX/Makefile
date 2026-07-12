all: sis3.pdf chris3.pdf bd3.pdf

PD=../../CosmoSimPy/

sis7.txt chris7.txt bd7.txt:
sis3.tex chris3.tex bd3.tex:

%.txt: %.raw
	sort $< > $@

sis%.raw:  $(PD)amplitudes.py
	time python3 $< --output $@ $* 8
sis%.tex:  $(PD)amplitudes.py
	time python3 $< --tex $@ $* 8

chris%.raw: $(PD)sieamplitudes.py 
	time python3 $< --lens SIS --output $@ $* 8

chris%.tex: $(PD)sieamplitudes.py 
	time python3 $< --lens SIS --tex $@ $* 8

bd%.raw: $(PD)sieorig.py 
	time python3 $< --lens SIS --output $@ $* 8

bd%.tex: $(PD)sieorig.py 
	time python3 $< --lens SIS --tex $@ $* 8


%.pdf: %.tex
	pdflatex $<

clean:
	rm -f *.tex *.pdf
		
