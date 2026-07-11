# (C) 2023: Hans Georg Schaathun <georg@schaathun.net> 

export PATH := ../bin:$(PATH)

all: alltests

status: 
	cat */*.status

alltests: 
	$(MAKE) images.status
	$(MAKE) compare.status

images.status:
	( makeimages.sh && \
		echo makeimages.sh succeeded > $@ || \
		echo makeimages.sh error: $$? > $@ ) 
compare.status:
	( diff `cat baseline.config`.csv roulette.csv && \
		echo `pwd`: compare succeeded > $@ || \
		echo `pwd`: discrepancy: $$? > $@ ) 

clean:
	rm -f *.status

