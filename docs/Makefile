all: lensmodel.svg

%.pdf: %.tex
	pdflatex $<

%.svg: %.pdf
	inkscape -z $< --export-plain-svg=$@
