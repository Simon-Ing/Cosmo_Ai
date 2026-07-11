#! /bin/sh

. ../../pythonenv/bin/activate


opt=$*

echo Options: "$*"

( sh clean.sh &&
  sh make.sh $opt ) | tee tutti.log
