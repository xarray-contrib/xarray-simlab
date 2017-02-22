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


## Correspondance between input file and cmd args

Look Jupyter configuration files and cmd args. Look also Ansible.
Cmd args always overwrite parameters in the input file.


## Set random seed

Needed for comparison between different experiments.
Allow setting the seed as a command-line argument and/or in the
input file.