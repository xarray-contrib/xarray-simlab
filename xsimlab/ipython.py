import textwrap

from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core import magic_arguments
import attr

from .formatting import format_var_dims
from .model import Model
from .utils import variables_dict


setup_template = """
import xsimlab as xs

ds_in = xs.create_setup(
    model={model},
    clocks={{}},
    input_vars={{
        {in_vars}
    }},
    output_vars={{}}
)
"""


def format_input_vars(
    model, skip_default=False, default=False, verbose=0, nested=False
):
    lines = []

    for pn, vnames in model.input_vars_dict.items():
        plines = []

        for vn in vnames:
            var = variables_dict(type(model[pn]))[vn]

            if skip_default and var.default is not attr.NOTHING:
                continue

            if default and var.default is not attr.NOTHING:
                default_val = f"{var.default!r}"
            else:
                default_val = ""

            comment = ""
            if verbose:
                var_desc = var.metadata["description"]
                comment += f"# {var_desc}\n" if var_desc else "# ---\n"
            if verbose > 1:
                var_dims = format_var_dims(var)
                if var_dims:
                    comment += f"#    dimensions: {var_dims}\n"
                if var.metadata["static"]:
                    comment += f"#    static variable: time/clock extra-dimension not allowed\n"
            if verbose > 2:
                var_attrs = var.metadata.get("attrs", False)
                if var_attrs:
                    for k, v in var_attrs.items():
                        comment += f"#    {k}: {v}\n"

            if nested:
                plines.append(comment + f"'{vn}': {default_val},")
            else:
                lines.append(comment + f"'{pn}__{vn}': {default_val},")

        if nested and plines:
            pfmt = textwrap.indent("\n".join(plines), " " * 4)
            lines.append(f"'{pn}': {{\n{pfmt}\n}},")

    return textwrap.indent("\n".join(lines), " " * 8)[8:]


@magics_class
class SimulationMagics(Magics):
    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument("model", help="xsimlab.Model object")
    @magic_arguments.argument(
        "-s",
        "--skip-default",
        action="store_true",
        default=False,
        help="Don't add input variables that have default values",
    )
    @magic_arguments.argument(
        "-d",
        "--default",
        action="store_true",
        default=False,
        help="Add input variables default values, if any (ignored if --skip-default)",
    )
    @magic_arguments.argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (i.e., add more input variables info as comments)",
    )
    @magic_arguments.argument(
        "-n",
        "--nested",
        action="store_true",
        default=False,
        help="Group input variables by process",
    )
    def create_setup(self, line=""):
        """Pre-fill the current cell with a new simulation setup."""

        args = magic_arguments.parse_argstring(self.create_setup, line)
        model_obj = self.shell.user_ns.get(args.model)

        if model_obj is None:
            raise KeyError(f"Model '{args.model}' not defined or not imported")
        elif not isinstance(model_obj, Model):
            raise TypeError(f"'{args.model}' is not a xsimlab.Model object")

        rendered = setup_template.format(
            model=args.model,
            in_vars=format_input_vars(
                model_obj,
                skip_default=args.skip_default,
                default=args.default,
                verbose=args.verbose,
                nested=args.nested,
            ),
        )

        content = f"# %create_setup {line}" + rendered

        self.shell.set_next_input(content, replace=True)


def load_ipython_extension(ipython):
    ipython.register_magics(SimulationMagics)
