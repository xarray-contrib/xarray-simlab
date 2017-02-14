# Discussion topics

## Options for accelerating the Python implementation

Using numpy were possible, but not always the best option since the
vectorized code may be harder to read and less efficient.

Three options for accelerating without vectorizing
(see benchmark notebook for avantages/limits of each):
- Numba
- Cython
- F2py

## Parallelization needs and strategies

Parallelization of fastscape is not straightforward:
mesh chunks = catchments/stacks = complex and heterogeneous geometries !

Parallelization of fastscape inversion with NA or another sampler
is straightforward.

Do we really need to run (now or in a near future) massive
fastscape simulation? Parallelization of fastscape: a dev priority?

## Additional, "structural" model features.

Will we ever add the features below to Fastscape? The design of
Fastscape implementation will depend strongly on this.

- Irregular meshes
- Dynamic meshes
- Flexible boundary conditions
- Alternative / dynamic parameterizations
- Multiple direction flow
- Third dimension (sedimentation...).

## Other features

- real-time monitoring/plotting.

## Name of the package?

"PyScape"?

## Warning: new project

"Start small and iterate by small steps"

You are alpha/beta users!!

-> public API may change often, annoying for (end)-users,
but lots of benefits in the long-term.

## License?

Very important!!!

GPLv3, LGPLv3 (copyleft) - or - MIT, BSD...

## Internal core functions

Suggestion of module organization:

- flow_direction
  - `__init__.py`  (import from `.api`)
  - `algos.py`  (internal functions, e.g., numba)
  - `api.py`    (public functions - API)
- node_ordering
  - `__init__.py`
  - `algos.py`
  - `api.py`
- ...
