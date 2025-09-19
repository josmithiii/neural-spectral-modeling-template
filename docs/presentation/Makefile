NAME = NSMT

#LATEX = pdflatex -shell-escape
LATEX = latex -shell-escape
FIGTEX2EPS = figtex2eps
#FIGTEX2EPS = figtex2eps -v

# Define PNG to EPS conversion variables
pngs  := $(wildcard png/*.png)
pngepss := $(pngs:png/%.png=eps/%.eps)

# Define JPG to EPS conversion variables
jpgs  := $(wildcard jpg/*.jpg)
jpgepss := $(jpgs:jpg/%.jpg=eps/%.eps)

# Use latexmk for building (dvips/PSTricks-friendly)
LATEXMK = latexmk
# Ghostscript invocation avoiding deprecated .setpdfwrite (escape $ for make)
LATEXMK_PS2PDF = -e '$$ps2pdf = "gs -q -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile=%D %S";'

tc tclean: clean
	-/bin/rm -f $(NAME)
	-/bin/rm -f $(NAME).pdf $(NAME).ps $(NAME).bm $(NAME).aux $(NAME).idx $(NAME).dvi $(NAME)-autopp.* *~

 clean:
	-$(LATEXMK) -C $(NAME).tex
	-/bin/rm -f $(NAME) $(NAME).blg $(NAME).bbl $(NAME).log $(NAME).nav $(NAME).aux $(NAME).snm $(NAME).toc $(NAME).out $(NAME).fdb_latexmk

# MUST USE $(LATEX) FOR LOGO
p pdf presentation: $(NAME).pdf

# Legacy DVI build (kept for reference)
$(NAME).dvi: $(NAME).tex *.tex pdf/*.pdf eps/*.eps fig/*.fig
	$(LATEX) $(NAME).tex
	@echo For paper only: bibtex $(NAME)
	@echo $(LATEX) $(NAME).tex
	$(LATEX) $(NAME).tex

$(NAME).pdf: $(NAME).tex *.tex pdf/*.pdf eps/*.eps fig/*.fig png/*.png $(pngepss) jpg/*.jpg $(jpgepss)
	$(LATEXMK) -pdfps -interaction=nonstopmode -halt-on-error $(LATEXMK_PS2PDF) $(NAME).tex

po: $(NAME).pdf
	open $(NAME).pdf

qp:
	$(LATEXMK) -pdfps -interaction=nonstopmode -halt-on-error $(LATEXMK_PS2PDF) $(NAME).tex

qpo:
	(make qp)
	open $(NAME).pdf

# Continuous preview
watch:
	$(LATEXMK) -pdfps -pvc -interaction=nonstopmode $(LATEXMK_PS2PDF) $(NAME).tex

pdfs := $(wildcard fir*plot.pdf H*.pdf)
crop:
	for p in $(pdfs) ; do \
		pdfcrop $$p $$p ; \
	done

figs  := $(wildcard fig/*.fig)
epss :=  $(figs:fig/%fig=eps/%eps)
epsfigs: $(epss)
#epss :=  $(patsubst fig/%.fig,eps/%.eps,$(figs))

eps/%.eps: fig/%.fig
	(cd ./fig; $(FIGTEX2EPS) $(<:fig/%=%))
	mv $(<:%.fig=%.eps) eps

pngeps: $(pngepss)

eps/%.eps: png/%.png
	convert $< $@

jpgeps: $(jpgepss)
eps/%.eps: jpg/%.jpg
	convert $< $@


ic installc:: $(NAME).pdf
	scp $(NAME).pdf jos@ccrma-gate.stanford.edu:~/Web/pdf

i install: $(NAME).pdf
	/bin/cp $(NAME).pdf /l/dttd/
	echo To install at CCRMA, say 'make ic'
