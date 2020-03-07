import sys, shutil
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

sys.argv[1:] = ["build_ext", "--inplace"]
setup(ext_modules = cythonize([Extension("ecco._ui",
                                         ["ecco/_ui.pyx"])],
                              language_level=3))
try :
    shutil.rmtree("build")
except :
    pass
