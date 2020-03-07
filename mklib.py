import os, os.path, sys, shutil, argparse
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

from ecoserv.utils import uptodate

parser = argparse.ArgumentParser("build.py")
parser.add_argument("-I", metavar="PATH", type=str,
                    default=[os.path.expanduser("~/.local/include")],
                    action="append",
                    help="libITS and libDDD source dir containing 'its/ITSModel.hh' "
                    "and 'ddd/DDD.h' (default '$HOME/.local/include')")
parser.add_argument("-P", metavar="PATH", type=str,
                    default=[os.path.expanduser("~/work/tools/its/pyddd"),
                             os.path.expanduser("~/work/tools/its/pyits")],
                    action="append",
                    help="pyddd and pyits source dir (default"
                    " '$HOME/work/tools/its/pyddd' and '$HOME/work/tools/its/pyits')")
args = parser.parse_args()

extensions = []

##
## ktzio
##

ktzio_ext = Extension("ecoserv.ktzio",
                      ["ecoserv/ktzio.pyx",
                       "ktzlib/ktzread.c",
                       "ktzlib/ktzfree.c"],
                      include_dirs = ["ktzlib", "/home/franck/.anaconda3/lib/python3.7/site-packages/numpy/core/include"],
                      libraries = ["z"],
                      library_dirs = [])

try :
    import ecoserv.ktzio as ktzio
    if uptodate ([ktzio.__file__], ["ecoserv/ktzio.pyx",
                                    "ktzlib/ktzread.c",
                                    "ktzlib/ktzfree.c",
                                    "ktzlib/ktz.h"]) :
        print("'ecoserv/%s' is up-to-date" % os.path.basename(ktzio.__file__))
        ktzio_ext = None
except :
    pass

if ktzio_ext is not None :
    extensions.append(ktzio_ext)

##
## build
##

if extensions :
    sys.argv[1:] = ["build_ext", "--inplace"]
    setup(ext_modules = cythonize(extensions,
                                  language_level=3,
                                  include_path=args.P))

##
## clean
##

try :
    shutil.rmtree("build")
except :
    pass
try :
    os.unlink("ecoserv/ktzio.c")
except :
    pass
