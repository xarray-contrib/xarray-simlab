.. _installing:

Install xarray-simlab
=====================

Required dependencies
---------------------

- Python 3.4 or later.
- `numpy <http://www.numpy.org/>`__
- `xarray <http://xarray.pydata.org>`__ (0.8.0 or later)

Optional dependencies
---------------------

For model visualization
~~~~~~~~~~~~~~~~~~~~~~~

- `graphviz <http://graphviz.readthedocs.io>`__

Install from source
-------------------

While there are ``xarray-simlab`` releases available at `PyPi <https://pypi.python.org/pypi/xarray-simlab/0.0.9>`_ 
and `conda-forge <https://github.com/conda-forge/xarray-simlab-feedstock>`_, you may prefer to install from source.

Be sure you have the required dependencies (numpy and xarray)
installed first. You might consider using conda_ to install them::

    $ conda install xarray numpy pip -c conda-forge

A good practice (especially for development purpose) is to install the packages
in a separate environment, e.g. using conda::

    $ conda create -n simlab_py36 python=3.6 xarray numpy pip -c conda-forge
    $ source activate simlab_py36

Then you can clone the ``xarray-simlab`` git repository and install it
using ``pip`` locally::

    $ git clone https://github.com/benbovy/xarray-simlab
    $ cd xarray-simlab
    $ pip install .

For development purpose, use the following command::

    $ pip install -e .

.. _PyPi: https://pypi.python.org/pypi
.. _conda-forge: https://conda-forge.github.io/
.. _conda: http://conda.io/

Import xarray-simlab
--------------------

To make sure that ``xarray-simlab`` is correctly installed, try to import it by
running this line::

    $ python -c "import xsimlab"
