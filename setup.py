#****************************************************#
# This file is part of OPTMOD                        #
#                                                    #
# Copyright (c) 2019, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTMOD is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension

ext_modules = cythonize([Extension(name='optmod.coptmod.coptmod',
                                   sources=['./optmod/coptmod/coptmod.pyx',
                                            './optmod/coptmod/evaluator.c',
                                            './optmod/coptmod/node.c'],
                                   libraries=[],
                                   include_dirs=[np.get_include(), './optmod/coptmod'],
                                   library_dirs=[],
                                   extra_link_args=[])])

setup(name='OPTMOD',
      zip_safe=False,
      version='0.0.1rc1',
      description='Optimization Modeling Library',
      url='',
      author='Tomas Tinoco De Rubira',
      author_email='ttinoco5687@gmail.com',
      include_package_data=True,
      license='BSD 2-Clause License',
      packages=['optmod',
                'optmod.coptmod'],
      install_requires=['cython>=0.20.1',
                        'numpy>=1.11.2',
                        'scipy>=0.18.1',
                        'optalg==1.1.8rc1',
                        'nose'],
      package_data={'optmod': []},
      classifiers=['Development Status :: 5 - Production/Stable',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.6'],
      ext_modules=ext_modules)
                                       
