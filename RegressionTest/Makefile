
all: alltests

clean alltests:
	cd exactpm ; $(MAKE) $@
	cd raytrace ; $(MAKE) $@
	cd rotatedsis ; $(MAKE) $@
	cd roulette ; $(MAKE) $@
	cd roulettepm ; $(MAKE) $@
baselines:
	cd exactpm ; $(MAKE) baseline`date "+%Y%m%d"`
	cd raytrace ; $(MAKE) baseline`date "+%Y%m%d"` 
	cd rotatedsis ; $(MAKE) baseline`date "+%Y%m%d"`
	cd roulette ; $(MAKE) baseline`date "+%Y%m%d"` 
	cd roulettepm ; $(MAKE) baseline`date "+%Y%m%d"`
