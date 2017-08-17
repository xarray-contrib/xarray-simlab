.. _faq:

Frequently Asked Questions
==========================

Does xarray-simlab provide built-in models?
-------------------------------------------

No, xarray-simlab provides only the framework for creating, customizing and
running computational models. It is intended to be a general-purpose tool.
Domain specific models should be implemented in 3rd party packages. For example,
`xarray-topo`_ provides xarray-simlab models and model components for simulating
landscape evolution.

.. _`xarray-topo`: https://gitext.gfz-potsdam.de/sec55-public/xarray-topo

Does xarray-simlab allow fast model execution?
----------------------------------------------

Yes, although it depends on how the model is implemented.

xarray-simlab is written in pure-Python and so is the outer (time) loop of
xarray-simlab models. The execution of Python code is slow compared to other
languages, but for the outer loop only it wouldn't represent the main bottleneck
of the overall model execution, especially when using an implicit time scheme.
For inner (e.g., spatial) loops in each model processes - implemented within
their ``.run_step`` method -, it might be better to have a numpy vectorized
implementation, use an accelerator like Cython_ or Numba_ or call wrapped code
that is written in, e.g., C or Fortran (see for example f2py_ for wrapping
Fortran code).

As with any other framework, xarray-simlab introduces an overhead compared to
a simple, straightforward (but non-flexible) implementation of a model. However,
the preliminary benchmarks that we have run show only a very small overhead.

.. _Cython: http://cython.org/
.. _Numba: http://numba.pydata.org/
.. _f2py: https://docs.scipy.org/doc/numpy-dev/f2py/

.. question_to_add : Does xarray-simlab allow model execution in parallel?

.. question_to_add: Can xarray-simlab be used without xarray?

.. question_to_add: I already have model implementation.

   Can I use xarray-simlab with it?  A: xarray-simlab encourages fine-grain
   processes, but it is easy to warp more monolitic code as process, add an
   interface (variables) and then leverage the features of xarray-simlab.

Will xarray-simlab support Python 2.7.x?
----------------------------------------

No, unless there are very good reasons to do so. The main packages of the Python
scientific ecosystem support Python 3.4 or later, and it seems that Python 2.x
will not be maintained anymore past 2020 (see `PEP 373`_). Although some tools
easily allow supporting both Python 2 and 3 versions in a single code base,
it still makes the code harder to maintain.

.. _`PEP 373`: https://www.python.org/dev/peps/pep-0373/


Which features are likely to be implemented in next xarray-simlab releases?
---------------------------------------------------------------------------

xarray-simlab is a very young project. Some ideas for future development can be
found in the roadmap_ on the xarray-simlab's Github wiki_.

.. _roadmap: https://github.com/benbovy/xarray-simlab/wiki/Roadmap
.. _wiki: https://github.com/benbovy/xarray-simlab/wiki
