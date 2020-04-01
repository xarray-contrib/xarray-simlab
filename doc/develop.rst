.. _develop:

Contributor Guide
=================

xarray-simlab is an open-source project. Contributions are welcome, and they are
greatly appreciated!

You can contribute in many ways, e.g., by reporting bugs, submitting feedbacks,
contributing to the development of the code and/or the documentation, etc.

This page provides resources on how best to contribute.

Issues
------

The `Github Issue Tracker`_ is the right place for reporting bugs and for
discussing about development ideas. Feel free to open a new issue if you have
found a bug or if you have suggestions about new features or changes.

For now, as the project is still very young, it is also a good place for
asking usage questions.

.. _`Github Issue Tracker`: https://github.com/benbovy/xarray-simlab/issues

Development environment
-----------------------

If you wish to contribute to the development of the code and/or the
documentation, here are a few steps for setting a development environment.

Fork the repository and download the code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To further be able to submit modifications, it is preferable to start by
forking the xarray-simlab repository on GitHub_ (you need to have an account).

Then clone your fork locally::

  $ git clone git@github.com:your_name_here/xarray-simlab.git

Alternatively, if you don't plan to submit any modification, you can clone the
original xarray-simlab git repository::

   $ git clone git@github.com:benbovy/xarray-simlab.git

.. _GitHub: https://github.com

Install
~~~~~~~

To install the dependencies, we recommend using the conda_ package manager with
the conda-forge_ channel. For development purpose, you might consider installing
the packages in a new conda environment::

  $ conda create -n xarray-simlab_dev python attrs numpy xarray -c conda-forge
  $ source activate xarray-simlab_dev

Then install xarray-simlab locally (in development mode) using ``pip``::

  $ cd xarray-simlab
  $ pip install -e .

.. _conda: http://conda.pydata.org/docs/
.. _conda-forge: https://conda-forge.github.io/

Run tests
~~~~~~~~~

To make sure everything behaves as expected, you may want to run
xarray-simlab's unit tests locally using the `pytest`_ package. You
can first install it with conda::

  $ conda install pytest -c conda-forge

Then you can run tests from the main xarray-simlab directory::

  $ pytest xsimlab --verbose

.. _pytest: https://docs.pytest.org/en/latest/

Contributing to code
--------------------

Below are some useful pieces of information in case you want to contribute
to the code.

Local development
~~~~~~~~~~~~~~~~~

Once you have setup the development environment, the next step is to create
a new git branch for local development::

  $ git checkout -b name-of-your-bugfix-or-feature

Now you can make your changes locally.

Submit changes
~~~~~~~~~~~~~~

Once you are done with the changes, you can commit your changes to git and
push your branch to your xarray-simlab fork on GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

(note: this operation may be repeated several times).

We you are ready, you can create a new pull request through the GitHub_ website
(note that it is still possible to submit changes after your created a pull
request).

Python versions
~~~~~~~~~~~~~~~

xarray-simlab supports Python versions 3.6 and higher. It is not compatible with
Python versions 2.x. We don't plan to make it compatible with Python 2.7.x.

Tests
~~~~~

xarray-simlab's uses unit tests extensively to make sure that every
part of the code behaves as we expect. Test coverage is required for
all code contributions.

Unit tests are written using `pytest`_ style (i.e., mostly using the ``assert``
statement directly) in various files located in the ``xsimlab/tests`` folder.
The file ``conftest.py`` defines some ``process`` decorated classes, ``Model``
objects and ``xarray.Dataset`` objects that can be used as fixtures for testing.

You can run the tests locally from the main xarray-simlab directory::

  $ pytest xsimlab --verbose

All the tests are also executed automatically on continuous integration
platforms on every push to every pull request on GitHub.

Docstrings
~~~~~~~~~~

Everything (i.e., classes, methods, functions...) that is part of the public API
should follow the numpydoc_ standard when possible.

.. _numpydoc: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

Code Formatting & linting
~~~~~~~~~~~~~~~~~~~~~~~~~

xarray-simlab uses black_ and flake8_ to ensure a consistent code format
throughout the project. Both of these tools can be installed with either
``conda`` or ``pip``. Once installed in your development environment, your can
run them from the root of the xarray-simlab repository::

   $ black .
   $ flake8

to auto-format your code. For convenience, many editors have plugins that will
apply ``black`` as you edit files.

``flake8`` reports warnings and/or errors about code formatting. It may also
detect other programming errors.

Like unit tests, These tools are also run on continuous platforms for every code
change submission.

.. _black: https://black.readthedocs.io/en/stable/
.. _flake8: http://flake8.pycqa.org

Release notes
~~~~~~~~~~~~~

Every significative code contribution should be listed in Section
:doc:`whats_new` of this documentation under the corresponding version.

Contributing to documentation
-----------------------------

xarray-simlab uses Sphinx_ for documentation, hosted on http://readthedocs.org .
Documentation is maintained in the RestructuredText markup language (``.rst``
files) in ``xarray-simlab/doc``.

To build the documentation locally, first install requirements (for example here
in a separate conda environment)::

   $ conda env create -n xarray-simlab_doc -f doc/environment.yml
   $ source activate xarray-simlab_doc

Then build documentation with ``make``::

   $ cd doc
   $ make html

The resulting HTML files end up in the ``build/html`` directory.

You can now make edits to rst files and run ``make html`` again to update
the affected pages.

.. _Sphinx: http://www.sphinx-doc.org/
