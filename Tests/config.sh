
echo $0
echo $1

rootdir=`( while test ! -d .git ; do cd .. ; done ; pwd )`
export PATH=$rootdir/CosmoSimPy:$rootdir/Scripts:$PATH

echo rootdir: $rootdir

if test x$OSTYPE = xcygwin -o x$OSTYPE = xmsys
then
   if which magick
   then 
      CONVERT="magick convert"
   fi
elif which convert
then
    CONVERT="convert"
elif which magick
then
    CONVERT="magick convert"
fi

echo ImageMagick command is: $CONVERT
$CONVERT --version
