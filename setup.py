#!/usr/bin/env python

from setuptools import setup, find_packages
from os.path import exists

import versioneer


setup(name='xarray-simlab',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description=('generic framework and xarray extension '
                   'for computer model simulations'),
      url='http://github.com/benbovy/xarray-simlab',
      maintainer='Benoit Bovy',
      maintainer_email='benbovy@gmail.com',
      license='BSD-Clause3',
      keywords='python xarray modelling simulation framework',
      packages=find_packages(),
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      python_requires='>=3.5',
      install_requires=['attrs >= 18.1.0', 'numpy', 'xarray >= 0.10.0'],
      tests_require=['pytest >= 3.3.0'],
      zip_safe=False)
