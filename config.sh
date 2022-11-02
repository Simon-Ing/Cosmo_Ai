module purge
module load intel/2020b

export PATH="$HOME/local/usr/sbin:$HOME/local/usr/bin:$HOME/local/bin:$PATH"
export MANPATH="$HOME/local/usr/share/man:$MANPATH"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/local/usr/lib:$HOME/local/usr/lib64"
export LD_RUN_PATH="$LD_RUN_PATH:$HOME/local/usr/lib:$HOME/local/usr/lib64"

export CPATH=$HOME/local/usr/include:$CPATH

rm -rf build
CWD=`pwd`
mkdir -p build
cd build
cmake ..
cd $CWD
# make 
