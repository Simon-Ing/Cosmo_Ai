
SVG=lensmodel.svg points.svg relativeeta.svg
PDF=lensmodel.pdf points.pdf relativeeta.pdf

all: $(SVG)
pdf: $(PDF)

$(PDF):

install: $(SVG)
	cp $(SVG) ../../cosmoai-aes.github.io/docs/model

%.pdf: %.tex
	pdflatex $<

%.svg: %.pdf
	inkscape --export-type=svg --pdf-poppler -l $<
