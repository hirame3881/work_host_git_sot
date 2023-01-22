FILENAME  = thesis
TEX       = $(FILENAME).tex
DVI       = $(FILENAME).dvi
all:
	make tex
	make bib
	make tex
	make tex
	make dvipdf
clean:
	-rm *.aux *.bbl *.bcf *.bib *.blg *.dvi *.idx *.ilg *.ind *.log *.out *.run.xml *.synctex.gz *.toc *~
tex:
	platex $(TEX)
bib:
	bibtex $(FILENAME)
dvipdf:
	dvipdfmx $(DVI)