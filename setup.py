#!/usr/bin/env python

from setuptools import setup
from os.path import exists

import versioneer


setup(name='xarray-simlab',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='xarray extension for computer model simulations',
      url='http://github.com/benbovy/xarray-simlab',
      maintainer='Benoit Bovy',
      maintainer_email='benbovy@gmail.com',
      license='BSD-Clause3',
      keywords='python xarray modelling simulation-framework',
      packages=['xsimlab', 'xsimlab.variable'],
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      python_requires='>=3.4',
      install_requires=['numpy', 'xarray >= 0.8.0'],
      zip_safe=False)
