
SVG=lensmodel.svg points.svg
all: $(SVG)

install: $(SVG)
	cp $(SVG) ../../cosmoai-aes.github.io/docs/model

%.pdf: %.tex
	pdflatex $<

%.svg: %.pdf
	inkscape -z $< --export-plain-svg=$@
