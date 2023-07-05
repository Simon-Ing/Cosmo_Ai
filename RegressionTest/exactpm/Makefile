# (C) 2023: Hans Georg Schaathun <georg@schaathun.net> 

all: 
	$(MAKE) images.status
	$(MAKE) compare.status
	$(MAKE) montage.status
montage.status:
	( montageimages.sh && \
		echo Montage error: $? > $@ || \
		echo Montage succeeded ) | tee mottnage.log
images.status:
	( makeimages.sh && \
		echo makeimages.sh error: $? > $@ || \
		echo makeimages.sh succeeded ) | tee images.log
compare.status:
	( compareimages.sh && \
		echo compareimages.sh error: $? > $@ || \
		echo compareimages.sh succeeded ) | tee compare.log
