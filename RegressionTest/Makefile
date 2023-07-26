
all: alltests

status: 
	cat */*.status

clean alltests:
	cd exactpm ; $(MAKE) $@
	cd raytrace ; $(MAKE) $@
	cd roulette ; $(MAKE) $@
	cd roulettepm ; $(MAKE) $@
	cd roulettelens ; $(MAKE) $@
	cd amplitudes ; $(MAKE) $@
	$(MAKE) status

baselines:
	cd exactpm ; $(MAKE) baseline`date "+%Y%m%d"`
	cd raytrace ; $(MAKE) baseline`date "+%Y%m%d"` 
	cd roulette ; $(MAKE) baseline`date "+%Y%m%d"` 
	cd roulettepm ; $(MAKE) baseline`date "+%Y%m%d"`
false:
	/bin/false | tee test.log ; echo $${PIPESTATUS[0]} $${PIPESTATUS[1]} 
true:
	/bin/true
15:
	exit 15
