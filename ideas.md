# Ideas


## xarray accessors

- fscape (for setup model runs)

- ftopo (for toolbox)

- fplot (for plotting, e.g., river profiles)?


## Organizing the documentation

1. xarray datamodel (load data from a netcdf resulting from a
   model run and show some useful dataset handling)
2. fscape extension
3. ftopo extension
4. fplot extension?


## `overwrite` kwarg to `set_` methods?

`set_` methods update the Dataset inplace. Maybe less error-prone
to add an overwrite argument that defaults to `False`.


## Multiple input files

Common Use case: a top-level working directory for a project
and sub-directories for different experiments that share common
settings.

1. Allow providing a list of yaml input filenames to
   `create_fastscape_setup`.

2. Allow providing references to other input files within an
   input file. In case of conflict, parameter values set in
   the current file will overwrite those set in the referance file
   (more like Ansible).

Option 2 seems the best.

## extend yaml syntax?

Option 1: inline spec like in Ansible

```
grid: nnodes=101, length=1e5
```

Option 2: indent

```
grid:
    nnodes: 101
    length: 1e5
```

Option 1 not supported by default with pyaml, but
more succint and closer to the API. Maybe allow both?
Do we want to follow strict yaml syntax so that input files
can be loaded using other tools / languages?

Look in Ansible code how pyaml is extended (custom Loader?)

## Set variables in input file

Like in Ansible. It can be useful, for example to reuse the same
value at different places (e.g., master clock and output clocks...)

Require Jinja templating


## Correspondance between input file and cmd args

Look Jupyter configuration files and cmd args. Look also Ansible.
Cmd args always overwrite parameters in the input file.


## Set random seed

Needed for comparison between different experiments.
Allow setting the seed as a command-line argument and/or in the
input file.