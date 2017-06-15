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

Some ideas for future development can be found in the roadmap_ on the
xarray-simlab's Github wiki_.

.. _roadmap: https://github.com/benbovy/xarray-simlab/wiki/Roadmap
.. _wiki: https://github.com/benbovy/xarray-simlab/wiki
