#****************************************************#
# This file is part of OPTMOD                        #
#                                                    #
# Copyright (c) 2019, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTMOD is released under the BSD 2-clause license. #
#*******#********************************************#

from setuptools import setup

setup(name='OPTMOD',
      zip_safe=False,
      version='0.0.1',
      description='Optimization Modeling Library',
      url='',
      author='Tomas Tinoco De Rubira',
      author_email='ttinoco5687@gmail.com',
      include_package_data=True,
      license='BSD 2-Clause License',
      packages=['optmod'],
      install_requires=['numpy>=1.11.2',
                        'scipy>=0.18.1',
                        'networkx==1.11',
                        'optalg==1.1.7rc1',
                        'nose'],
      package_data={'optmod': []},
      classifiers=['Development Status :: 5 - Production/Stable',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5'])
                                       
