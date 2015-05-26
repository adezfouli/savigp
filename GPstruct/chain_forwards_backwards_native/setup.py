from distutils.core import setup, Extension
import numpy

# define the extension module
chain_forwards_backwards_native = Extension('chain_forwards_backwards_native', sources=['chain_forwards_backwards_native.c'],
                          include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[chain_forwards_backwards_native])
