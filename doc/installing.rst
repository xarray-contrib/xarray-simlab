.. _installing:

Install xarray-simlab
=====================

Required dependencies
---------------------

- Python 3.6 or later.
- `attrs <http://www.attrs.org>`__ (18.2.0 or later)
- `numpy <http://www.numpy.org/>`__
- `xarray <http://xarray.pydata.org>`__ (0.10.0 or later)
- `zarr <https://zarr.readthedocs.io>`__ (2.3.0 or later)
- `dask <https://docs.dask.org>`__

Optional dependencies
---------------------

- `graphviz <http://graphviz.readthedocs.io>`__ (for model visualization)

Install using conda
-------------------

xarray-simlab can be installed or updated using conda_::

  $ conda install xarray-simlab -c conda-forge

This installs xarray-simlab and all common dependencies, including
numpy and xarray.

xarray-simlab conda package is maintained on the `conda-forge`_
channel.

.. _conda-forge: https://conda-forge.org/
.. _conda: https://conda.io/docs/

Install using pip
-----------------

You can also install xarray-simlab and its required dependencies using
``pip``::

  $ pip install xarray-simlab

Install from source
-------------------

To install xarray-simlab from source, be sure you have the required
dependencies (numpy and xarray) installed first. You might consider
using conda_ to install them::

    $ conda install attrs xarray dask zarr numpy pip -c conda-forge

A good practice (especially for development purpose) is to install the
packages in a separate environment, e.g. using conda::

    $ conda create -n xsimlab python attrs xarray dask zarr numpy pip -c conda-forge
    $ source activate xsimlab

Then you can clone the xarray-simlab git repository and install it
using ``pip`` locally::

    $ git clone https://github.com/benbovy/xarray-simlab
    $ cd xarray-simlab
    $ pip install .

For development purpose, use the following command::

    $ pip install -e .

.. _PyPi: https://pypi.python.org/pypi/xarray-simlab/

Import xarray-simlab
--------------------

To make sure that xarray-simlab is correctly installed, try to import
it by running this line::

    $ python -c "import xsimlab"
