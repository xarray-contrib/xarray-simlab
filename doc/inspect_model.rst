.. _inspect_model:

Inspect Models
==============

We can inspect xarray-simlab's :class:`~xsimlab.Model` objects in
different ways. As an example we'll use here the object ``model2``
which has been created in the previous section :doc:`create_model` of
this user guide.

.. ipython:: python
   :suppress:

    import sys
    sys.path.append('scripts')
    from advection_model import model2

Inspect processes and variables
-------------------------------

Model *repr* already gives information about the number and names of
processes and their variables that need an input value (if provided, a
short description is also displayed for these variables):

.. ipython:: python

    model2

For deeper inspection, Model objects support both dict-like and
attribute-like access to their processes, e.g.,

.. ipython:: python

    model2['advect']
    model2.grid

As shown here above, process *repr* includes:

- the path to the corresponding Process subclass (top line) ;
- a "Variables" section with all variables declared in the process
  (not only model inputs), their types (e.g., ``FloatVariable``,
  ``ForeignVariable``) and some additional information depending on
  their type like dimension labels for regular variables or references
  to original variables for ``ForeignVariable`` objects ;
- a "Meta" section with all process metadata.

In "Variables" section, symbol ``*`` means that a value for the
variable is provided by the process itself (i.e., ``provided=True``).

Processes also support dict-like and attribute-like access to all
their declared variables, e.g.,

.. ipython:: python

    model2['advect']['v']
    model2.grid.x

It is also possible to have direct access to all input variables in a
model using the :attr:`~xsimlab.Model.input_vars` property.

We can further test whether a variable is a model input or not, e.g.,

.. ipython:: python

    model2.is_input(('advect', 'v'))
    model2.is_input(('profile', 'u'))

Visualize models as graphs
--------------------------

.. ipython:: python
   :suppress:

    from xsimlab.dot import dot_graph
    dot_graph(model2, filename='savefig/model2_simple.png')
    dot_graph(model2, show_inputs=True, filename='savefig/model2_inputs.png')
    dot_graph(model2, show_inputs=True, show_variables=True,
              filename='savefig/model2_variables.png')

.. ipython:: python
   :suppress:

    dot_graph(model2, show_only_variable=('profile', 'u'),
              filename='savefig/model2_var_u.png')

It is possible to visualize a model and its processes as a directed
graph (note: this requires installing Graphviz and its Python
bindings, which both can be found on conda-forge):

.. ipython:: python

    model2.visualize();

.. image:: savefig/model2_simple.png
   :width: 40%

``show_inputs`` option allows to show model input variables as yellow
square nodes linked to their corresponding processes:

.. ipython:: python

    model2.visualize(show_inputs=True);

.. image:: savefig/model2_inputs.png
   :width: 60%

``show_variables`` option allows to show the other variables as white
square nodes:

.. ipython:: python

    model2.visualize(show_inputs=True, show_variables=True);

.. image:: savefig/model2_variables.png
   :width: 60%

Nodes with solid border correspond to regular variables while nodes
with dashed border correspond to ``ForeignVariable`` objects. 3d-box
nodes correspond to iterables of Variable objects, like
``VariableGroup``. Variables connected to their process with an arrow
have a value provided by the process.

A third option ``show_only_variable`` allows to show only one given
variable and all its references in other processes, e.g.,

.. ipython:: python

    model2.visualize(show_only_variable=('profile', 'u'));

.. image:: savefig/model2_var_u.png
   :width: 40%

Note that there is another function ``dot_graph`` available in module
``xsimlab.dot`` which produces similar graphs and which has a few more
options.
