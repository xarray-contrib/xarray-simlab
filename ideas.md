# Ideas


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


## Correspondance between input file and cmd args

Look Jupyter configuration files and cmd args. Look also Ansible.
Cmd args always overwrite parameters in the input file.


## Set random seed

Needed for comparison between different experiments.
Allow setting the seed as a command-line argument and/or in the
input file.