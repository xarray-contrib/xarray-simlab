.. _testing:

Testing
=======

Testing and/or debugging the logic implemented in process classes can
be achieved easily just by instantiating them. The xarray-simlab
framework is not invasive and process classes can be used like other,
regular Python classes.

.. ipython:: python
   :suppress:

    import sys
    sys.path.append('scripts')
    from advection_model import InitUGauss

Here is an example with the ``InitUGauss`` process class created in Section
:doc:`create_model`:

.. ipython:: python

    import numpy as np
    import matplotlib.pyplot as plt
    gauss = InitUGauss(loc=0.3, scale=0.1, x=np.arange(0, 1.5, 0.01))
    gauss.initialize()
    @savefig gauss.png width=50%
    plt.plot(gauss.x, gauss.u);

Like for any other process class, the parameters of the
``InitUGauss`` constructor correspond to each of the variables declared
in that class with either ``intent='in'`` or ``intent='inout'``. Those
parameters are "keyword only" (see `PEP 3102`_), i.e., it is not
possible to set these as positional arguments.

.. _`PEP 3102`: https://www.python.org/dev/peps/pep-3102/
