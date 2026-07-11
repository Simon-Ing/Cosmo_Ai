#! /bin/sh

. ../../venv/test/bin/activate

opt=$*

echo Options: "$*"

( sh clean.sh &&
  sh sis.sh $opt &&
  sh compare.sh ) | tee tutti.log
