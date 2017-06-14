.. _about:

About xarray-simlab
===================



Motivation
----------

The project is being developped with the idea of

Sources of inspiration
----------------------

xarray: data structure
dask: task graphs (DAGs)
luigi: use classes as primary, re-usable units for chaining many tasks
django: especially The Django's ORM part (i.e., django.db.models) for the
design of Process and Variable classes
param: another source of inspiration for Process interface and Variable objects.
climlab: another python package for process-oriented modeling, applied to
climate


(put this in create models section)

Variable.state (or value, rate, change) should not be used (get/set) outside
of Process subclasses.

ForeignVariable.state return the same object (usually a numpy array) than
Variable.state (replace class names by variable names in processes).
ForeignVariable.state is actually a shortcut to ForeignVariable.ref_var.state.
