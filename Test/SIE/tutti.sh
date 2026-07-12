#! /bin/sh

. ../../pythonenv/bin/activate


opt=$*

echo Options: "$*"

( sh clean.sh &&
  sh sie.sh $opt &&
  sh compare.sh ) | tee tutti.log
