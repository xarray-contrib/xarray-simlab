.. _inspect_model:

Model introspection
===================

.. ipython:: python
   :suppress:

    import sys
    sys.path.append('scripts')
    from advection_model import model2


.. ipython:: python

   model2


.. ipython:: python

   model2.advect
   model2['advect']

.. ipython:: python

   model2['advect']['v']
   model2.advect.v

.. ipython:: python

   model2.visualize()
