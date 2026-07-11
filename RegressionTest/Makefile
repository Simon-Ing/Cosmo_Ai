
all: alltests

status: cpptests
	cat */*.status

clean alltests:
	cd raytrace ; $(MAKE) $@
	cd roulette ; $(MAKE) $@
	cd old-sampled ; $(MAKE) $@
	cd new-roulette ; $(MAKE) $@
	cd new-raytrace ; $(MAKE) $@
	cd sampled-roulette ; $(MAKE) $@
	cd sampled-raytrace ; $(MAKE) $@
	cd cluster-roulette ; $(MAKE) $@
	cd cluster-raytrace ; $(MAKE) $@
	cd cluster-sampled-roulette ; $(MAKE) $@
	cd cluster-sampled-raytrace ; $(MAKE) $@
	cd roulettegen-old ; $(MAKE) $@
	cd roulettegen ; $(MAKE) $@
	cd sersic ; $(MAKE) $@
	cd rouletteoutput ; $(MAKE) $@
	cd amplitudes ; $(MAKE) $@

cpptests:
	cd raytrace ; $(MAKE) alltests
	cd roulette ; $(MAKE) alltests
	cd old-sampled ; $(MAKE) alltests
	cd new-roulette ; $(MAKE) alltests
	cd new-raytrace ; $(MAKE) alltests
	cd sampled-roulette ; $(MAKE) alltests
	cd sampled-raytrace ; $(MAKE) alltests
	cd cluster-roulette ; $(MAKE) alltests
	cd cluster-raytrace ; $(MAKE) alltests
	cd cluster-sampled-roulette ; $(MAKE) alltests
	cd cluster-sampled-raytrace ; $(MAKE) alltests
	cd roulettegen-old ; $(MAKE) alltests
	cd roulettegen ; $(MAKE) alltests
	cd sersic ; $(MAKE) $@
	cd rouletteoutput ; $(MAKE) $@

baselines:
	cd raytrace ; $(MAKE) baseline`date "+%Y%m%d"` 
	cd roulette ; $(MAKE) baseline`date "+%Y%m%d"` 
	cd old-sampled ; $(MAKE) baseline`date "+%Y%m%d"` 
	cd new-roulette ; $(MAKE)  baseline`date "+%Y%m%d"`
	cd new-raytrace ; $(MAKE)  baseline`date "+%Y%m%d"`
	cd sampled-roulette ; $(MAKE)  baseline`date "+%Y%m%d"`
	cd sampled-raytrace ; $(MAKE)  baseline`date "+%Y%m%d"`
	cd cluster-roulette ; $(MAKE)  baseline`date "+%Y%m%d"`
	cd cluster-raytrace ; $(MAKE)  baseline`date "+%Y%m%d"`
	# cd cluster-sampled-roulette ; $(MAKE)  baseline`date "+%Y%m%d"`
	# cd cluster-sampled-raytrace ; $(MAKE)  baseline`date "+%Y%m%d"`
	cd roulettegen-old ; $(MAKE)  baseline`date "+%Y%m%d"`
	cd roulettegen ; $(MAKE)  baseline`date "+%Y%m%d"`
false:
	/bin/false | tee test.log ; echo $${PIPESTATUS[0]} $${PIPESTATUS[1]} 
true:
	/bin/true
15:
	exit 15
