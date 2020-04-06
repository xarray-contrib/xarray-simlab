.. _faq:

Frequently Asked Questions
==========================

Does xarray-simlab provide built-in models?
-------------------------------------------

No, xarray-simlab provides only the framework for creating,
customizing and running computational models. It is intended to be a
general-purpose tool.  Domain specific models should be implemented in
3rd party packages. For example, `fastscape`_ provides xarray-simlab
models and model components for simulating landscape evolution.

.. _`fastscape`: https://github.com/fastscape-lem/fastscape

Can xarray-simlab be used with existing model implementations?
--------------------------------------------------------------

Yes, it should be easy to wrap existing model implementations using
xarray-simlab. Even monolithic codes may leverage the xarray
interface.  However, as the framework works best at a fine grained
level (i.e., with models built from many "small" components) it might
be worth to refactor those monolithic implementations.

Does xarray-simlab allow fast model execution?
----------------------------------------------

Yes, although it depends on how the model is implemented.

xarray-simlab is written in pure-Python and so is the outer (time)
loop in simulations. The execution of Python code is slow compared to
other languages, but for the outer loop only it wouldn't represent the
main bottleneck of the overall model execution, especially when using
an implicit time scheme. For inner (e.g., spatial) loops in each model
processes, it might be better to have a numpy vectorized
implementation, use tools like Cython_ or Numba_ or call wrapped code
that is written in, e.g., C/C++ or Fortran (see for example f2py_ for
wrapping Fortran code or pybind11_ for wrapping C++11 code).

As with any other framework, xarray-simlab introduces an overhead
compared to a simple, straightforward (but non-flexible)
implementation of a model. The preliminary benchmarks that we have run
show only a very small (almost free) overhead, though. This overhead
is mainly introduced by the thin object-oriented layer that model
components (i.e., Python classes) together form.

.. _Cython: http://cython.org/
.. _Numba: http://numba.pydata.org/
.. _f2py: https://docs.scipy.org/doc/numpy-dev/f2py/
.. _pybind11: https://pybind11.readthedocs.io

Does xarray-simlab support running model(s) in parallel?
--------------------------------------------------------

Yes! Three levels of parallelism are possible:

- "multi-models" parallelism, i.e., execution of multiple model runs in
  parallel,
- "single-model" parallelism, i.e., execution of multiple processes of
  a model in parallel,
- "user-specific" parallelism, i.e., parallel execution of some code
  written in one or more processes.

Note that the notion of process used above is different from multiprocessing: a
process here corresponds to a component of a model. See Section
:ref:`framework`.

For the first two levels, see Section :ref:`run_parallel`.

The third level "user-specific" is not part the xarray-simlab framework. Users
are free to develop xarray-simlab compatible models with custom code (in
processes) that is executed either sequentially or in parallel.

Is it possible to use xarray-simlab without xarray?
---------------------------------------------------

Although it sounds a bit odd given the name of this package, in
principle it is possible. The implementation of the modeling framework
is indeed completely decoupled from the xarray interface.

However, the xarray extension provided in this package aims to be the
primary, full-featured interface for setting and running simulations
from within Python.

The modeling framework itself doesn't have any built-in interface
apart from a few helper functions for running specific stages of a
simulation. Any other interface has to be built from scratch, but in
many cases it wouldn't require a lot of effort. In the future, we plan
to also provide an experimental interface for real-time, interactive
simulation based on tools like `ipywidgets`_, `bokeh`_ and/or
`holoviews`_.

.. _ipywidgets: https://github.com/jupyter-widgets/ipywidgets
.. _bokeh: https://github.com/bokeh/bokeh
.. _holoviews: https://github.com/ioam/holoviews

Will xarray-simlab support Python 2.7.x?
----------------------------------------

No, unless there are very good reasons to do so. The main packages of
the Python scientific ecosystem support Python 3.x, and it seems that
Python 2.x will not be maintained anymore past 2020 (see `PEP
373`_). Although some tools easily allow supporting both Python 2 and
3 versions in a single code base, it still makes the code harder to
maintain.

.. _`PEP 373`: https://www.python.org/dev/peps/pep-0373/


Which features are likely to be implemented in next xarray-simlab releases?
---------------------------------------------------------------------------

xarray-simlab is a very young project. Some ideas for future
development can be found in the roadmap_ on the xarray-simlab's Github
wiki_.

.. _roadmap: https://github.com/benbovy/xarray-simlab/wiki/Roadmap
.. _wiki: https://github.com/benbovy/xarray-simlab/wiki
