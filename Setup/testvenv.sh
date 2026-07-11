# (C) 2025: Hans Georg Schaathun <hg@schaathun.net>

D=/tmp/venv/test

mkdir -p $D

if test -r $D/bin/activate
then
   . $D/bin/activate
else
   sh Setup/build.sh

   mkdir -p venv
   python -m venv $D
   . $D/bin/activate

   pip install jupytext jupyterlab 
   pip install -e .
fi
