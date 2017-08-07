import numpy as np
import xarray as xr


def _get_dims_from_variable(array, variable):
    """Given an array of values (snapshot) and a (xarray-simlab) Variable
    object, Return dimension labels for the array."""
    for dims in variable.allowed_dims:
        if len(dims) == array.ndim:
            return dims
    return tuple()


class DatasetModelInterface(object):
    """Interface between xarray.Dataset and Model.

    It is used to:

    - set model inputs using the variables of a Dataset object,
    - run model simulation stages,
    - take snapshots for given model variables (defined in attributes of
      Dataset) following one or several clocks (i.e., Dataset coordinates),
    - convert the snapshots back into xarray.Variable objects and return a
      new xarray.Dataset object.

    """
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        self.dim_master_clock = dataset.xsimlab.dim_master_clock
        if self.dim_master_clock is None:
            raise ValueError("missing master clock dimension / coordinate ")

    def init_snapshots(self):
        """Initialize snapshots for model variables given in attributes of
        Dataset.
        """
        self.snapshot_vars = self.dataset.xsimlab.snapshot_vars
        self.snapshot_values = {}
        for vars in self.snapshot_vars.values():
            self.snapshot_values.update({v: [] for v in vars})

        self.snapshot_save = {
            clock: np.in1d(self.dataset[self.dim_master_clock].values,
                           self.dataset[clock].values)
            for clock in self.snapshot_vars if clock is not None
        }

    def take_snapshot_var(self, key):
        """Take a snapshot of a given model variable (i.e., a copy of the value
        of its `state` property).
        """
        proc_name, var_name = key
        model_var = self.model._processes[proc_name]._variables[var_name]
        self.snapshot_values[key].append(np.array(model_var.state))

    def take_snapshots(self, istep):
        """Take snapshots at a given step index."""
        for clock, vars in self.snapshot_vars.items():
            if clock is None:
                if istep == -1:
                    for key in vars:
                        self.take_snapshot_var(key)
            elif self.snapshot_save[clock][istep]:
                for key in vars:
                    self.take_snapshot_var(key)

    def snapshot_to_xarray_variable(self, key, clock=None):
        """Convert snapshots taken for a specific model variable to an
        xarray.Variable object.
        """
        proc_name, var_name = key
        variable = self.model._processes[proc_name]._variables[var_name]

        array_list = self.snapshot_values[key]
        first_array = array_list[0]

        if len(array_list) == 1:
            data = first_array
        else:
            data = np.stack(array_list)

        dims = _get_dims_from_variable(first_array, variable)
        if clock is not None:
            dims = (clock,) + dims

        attrs = variable.attrs.copy()
        attrs['description'] = variable.description

        return xr.Variable(dims, data, attrs=attrs)

    def get_output_dataset(self):
        """Build a new output Dataset from the input Dataset and
        all snapshots taken during a model run.
        """
        from .xr_accessor import SimlabAccessor

        xr_variables = {}

        for clock, vars in self.snapshot_vars.items():
            for key in vars:
                var_name = '__'.join(key)
                xr_variables[var_name] = self.snapshot_to_xarray_variable(
                    key, clock=clock
                )

        out_ds = self.dataset.update(xr_variables, inplace=False)

        for clock in self.snapshot_vars:
            if clock is None:
                attrs = out_ds.attrs
            else:
                attrs = out_ds[clock].attrs
            attrs.pop(SimlabAccessor._snapshot_vars_key)

        return out_ds

    def check_model_inputs_in_dataset(self):
        """Check if all model inputs have their corresponding data variables
        in Dataset.
        """
        missing_data_vars = []

        for proc_name, vars in self.model.input_vars.items():
            for var_name, var in vars.items():
                xr_var_name = proc_name + '__' + var_name
                if xr_var_name not in self.dataset.data_vars:
                    missing_data_vars.append(xr_var_name)

        if missing_data_vars:
            raise KeyError("missing data variables %s in Dataset"
                           % missing_data_vars)

    def set_model_inputs(self, dataset):
        """Set model inputs values from a given Dataset object (may be a subset
        of self.dataset)."""
        for proc_name, vars in self.model.input_vars.items():
            for var_name, var in vars.items():
                xr_var_name = proc_name + '__' + var_name
                xr_var = dataset.get(xr_var_name)
                if xr_var is not None:
                    var.value = xr_var.values.copy()

    def split_data_vars_clock(self):
        """Separate in Dataset between data variables that have the master clock
        dimension and those that don't.
        """
        ds_clock = self.dataset.filter(
            lambda v: self.dim_master_clock in v.dims
        )
        ds_no_clock = self.dataset.filter(
            lambda v: self.dim_master_clock not in v.dims
        )
        return ds_clock, ds_no_clock

    @property
    def time_step_lengths(self):
        """Return a DataArray with time-step durations."""
        clock_coord = self.dataset[self.dim_master_clock]
        return clock_coord.diff(self.dim_master_clock).values

    def run_model(self):
        """Run the model.

        The is the main function of the interface. It set model inputs
        from the input Dataset, run the simulation stages one after
        each other, possibly sets time-dependent values provided for
        model inputs (if any) before each time step, take snaphots
        between the 'run_step' and the 'finalize_step' stages, and
        finally returns a new Dataset with all the inputs and the
        snapshots.

        """
        self.check_model_inputs_in_dataset()
        ds_clock, ds_no_clock = self.split_data_vars_clock()
        ds_clock_any = bool(ds_clock.data_vars)

        self.init_snapshots()
        self.set_model_inputs(ds_no_clock)
        self.model.initialize()

        for istep, dt in enumerate(self.time_step_lengths):
            if ds_clock_any:
                ds_step = ds_clock.isel(**{self.dim_master_clock: istep})
                self.set_model_inputs(ds_step)

            self.model.run_step(dt)
            self.take_snapshots(istep)
            self.model.finalize_step()

        self.take_snapshots(-1)
        self.model.finalize()

        return self.get_output_dataset()
