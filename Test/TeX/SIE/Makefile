all: chris3.pdf bd3.pdf

PD=../../../CosmoSimPy/

chris3.tex bd3.tex:

%.txt: %.raw
	sort $< > $@

chris%.raw: $(PD)sieamplitudes.py 
	time python3 $< --lens SIE --output $@ $* 8

chris%.tex: $(PD)sieamplitudes.py 
	time python3 $< --lens SIE --tex $@ $* 8

bd%.raw: $(PD)sieorig.py 
	time python3 $< --lens SIE --output $@ $* 8

bd%.tex: $(PD)sieorig.py 
	time python3 $< --lens SIE --tex $@ $* 8


%.pdf: %.tex
	pdflatex $<

clean:
	rm -f *.tex *.pdf *.aux *.log
		
