#!/bin/bash
ml python/2.7.13
ml py-numpy/1.14.3_py27
ml py-scipy/1.1.0_py27
ml viz
ml py-matplotlib/2.2.2_py27
#ml boost/1.69.0

# See this stackoverflow question
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
# for the magic in this command
SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#
# Base package root. All the other releavant folders are relative to this
# location.
#
export GPDSWROOT=$SETUP_DIR
echo "GPDSWROOT set to " $GPDSWROOT

#
# Add the root folder to the $PYTHONPATH so that we can effectively import
# the relevant (both SWIG and pure-Python) modules.
#
export PYTHONPATH=$GPDSWROOT:$PYTHONPATH
echo "PYTHONPATH set to " $PYTHONPATH

#
# Add the bin folder to the $PATH environmental variable.
#
export PATH=$GPDSWROOT/bin:$PATH

#
# SWIG stuff.
#

# This indicates the library folder for the SWIG Makefile and interface files
# and is used in two different ways:
# * in the base Makefile $GPDSWSWIG/swig.km the option -I${GPDSWSWIG} is
#   passed to the swig command so that all the module interface files
#   can include files from $GPDSWSWIG (e.g., $GPDSWSWIG/gpdsw.i)
# * in all the modules Makefile the basic Makefile is called via
#   include $GPDSWSWIG/swig.mk
export GPDSWSWIG=$GPDSWROOT/swig

# This is the path to the file with the all the basic common Python machinery
# to be used in all the modules setup.py files wi
# swig = imp.load_source('swig', os.environ['GPDSWSWIGPY'])
export GPDSWSWIGPY=$GPDSWROOT/swig/swig.py

# This is the folder where all the SWIG Python modules are created, i.e.,
# assuming that $GPDSWROOT is in the $PYTHONPATH, you do import them by doing
# import gpdswswig.xxx.yyy
export GPDSWPYMODULEDIR=$GPDSWROOT/gpdswswig

#
# This is the folder where the custom icons for the GUIs live---it is
# important to be able to load them at runtime.
#
export GPDSWGUIICONDIR=$GPDSWROOT/Gui/icons

#
# And this is the place where the local repository with the static html
# doxygen documentation is located.
# FIXME: we need a more robust mechanism than just $SETUP_DIR/../
#
export GPDSWDOCGIT=$SETUP_DIR/../ixpesw.bitbucket.org

#
# Source the setup file in the external lib, that gives access to all the
# include and library folders for the libraries themselves.
# If $GPDEXTROOT does not exists, we assume that it's sitting the filesystem
# right next to $GPDSWROOT.
#
#if [ -z "$GPDEXTROOT" ]; then
export GPDEXTROOT=$SETUP_DIR/../gpdext
#fi
source $GPDEXTROOT/setup.sh
echo "GPDEXTROOT set to " $GPDEXTROOT

#
# LD/DYLD_LIBRARY PATH---here we need to make distinction between Mac OS and
# Linux.
#
LIBPATH=$GPDSWPYMODULEDIR:$BOOST_LIBS:$CFITSIO_LIBS:$GEANT_LIBS:$QCUSTOMPLOT_LIBS:$CAEN_LIBS
if [ $(uname) == "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$LIBPATH:$DYLD_LIBRARY_PATH
    echo "Darwin architecture, DYLD_LIBRARY_PATH set to" $DYLD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$LIBPATH:$LD_LIBRARY_PATH
    echo "LD_LIBRARY_PATH set to " $LD_LIBRARY_PATH
fi

#
# Some Geant 4 stuff. This is essentially copied over from $GEANT_BIN/geant4.sh,
# so that we do not have to source it.
# Note that $GEANT_LIBS is appended to $LD_LIBRARY_PATH in the previous block,
# wich takes care of the line
# export LD_LIBRARY_PATH="`cd $geant4_envbindir/../lib64 > /dev/null ; pwd`":${LD_LIBRARY_PATH}
# in the aforementioned $GEANT_BIN/geant4.sh file.
#
export PATH=$GEANT_BIN:$PATH
export GEANT_DATA=$GEANT_ROOT/data
export G4LEDATA=$GEANT_DATA/G4EMLOW6.50
export G4ENSDFSTATEDATA=$GEANT_DATA/G4ENSDFSTATE2.1

