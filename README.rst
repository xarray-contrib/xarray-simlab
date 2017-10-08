xarray-simlab: xarray extension for computer model simulations
==============================================================

|Build Status| |Coverage| |Doc Status| |Zenodo|

xarray-simlab is a Python library that provides both a generic
framework for building computational models in a modular fashion and a
xarray_ extension for setting and running simulations using the
xarray's ``Dataset`` structure. It is designed for interactive and
exploratory modeling.

**Warning: this package is currently under heavy development (no stable release yet).**

.. _xarray: http://xarray.pydata.org
.. |Build Status| image:: https://travis-ci.org/benbovy/xarray-simlab.svg?branch=master
   :target: https://travis-ci.org/benbovy/xarray-simlab
   :alt: Build Status
.. |Coverage| image:: https://coveralls.io/repos/github/benbovy/xarray-simlab/badge.svg?branch=master
   :target: https://coveralls.io/github/benbovy/xarray-simlab?branch=master
   :alt: Coverage Status
.. |Doc Status| image:: http://readthedocs.org/projects/xarray-simlab/badge/?version=latest
   :target: http://xarray-simlab.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |Zenodo| image:: https://zenodo.org/badge/93938479.svg
   :target: https://zenodo.org/badge/latestdoi/93938479
   :alt: Citation

Documentation
-------------

A draft of documentation is hosted on ReadTheDocs:
http://xarray-simlab.readthedocs.io

License
-------

3-clause ("Modified" or "New") BSD license,
see `License file <https://github.com/benbovy/xarray-simlab/blob/master/LICENSE>`__.

xarray-simlab uses short parts of the code of the xarray_, pandas_ and
dask_ libraries. Their licenses are reproduced in the "licenses"
directory.

.. _pandas: http://pandas.pydata.org/
.. _dask: http://dask.pydata.org

Acknowledgment
--------------

This project is supported by the `Earth Surface Process Modelling`_
group of the GFZ Helmholtz Centre Potsdam.

.. _`Earth Surface Process Modelling`: http://www.gfz-potsdam.de/en/section/earth-surface-process-modelling/

Citation
--------

If you use xarray-simlab in a scientific publication, we would
appreciate a `citation`_.

.. _`citation`: http://xarray-simlab.readthedocs.io/en/latest/citation.html
