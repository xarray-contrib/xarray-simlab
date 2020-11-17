import textwrap

from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core import magic_arguments
import attr

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


def format_input_vars(model, default=False, comment=False, nested=False):
    lines = []

    for pn, vnames in model.input_vars_dict.items():
        plines = []

        for vn in vnames:
            var = variables_dict(type(model[pn]))[vn]

            if default and var.default is not attr.NOTHING:
                default_val = f"{var.default!r}"
            else:
                default_val = ""

            if comment:
                var_desc = var.metadata["description"]
                description = f"# {var_desc}\n" if var_desc else "# ---\n"
            else:
                description = ""

            if nested:
                plines.append(description + f"'{vn}': {default_val},")
            else:
                lines.append(description + f"'{pn}__{vn}': {default_val},")

        if nested:
            pfmt = textwrap.indent("\n".join(plines), " " * 4)
            lines.append(f"'{pn}': {{\n{pfmt}\n}},")

    return textwrap.indent("\n".join(lines), " " * 8)[8:]


@magics_class
class SimulationMagics(Magics):
    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument("model", help="xsimlab.Model object")
    @magic_arguments.argument(
        "-d",
        "--default",
        action="store_true",
        default=False,
        help="Add input variables default values (if defined)",
    )
    @magic_arguments.argument(
        "-c",
        "--comment",
        action="store_true",
        default=False,
        help="Add input variables descriptions as comments",
    )
    @magic_arguments.argument(
        "-n",
        "--nested",
        action="store_true",
        default=False,
        help="group input variables by process",
    )
    def create_setup(self, line=""):
        """Pre-fill the current cell with a new simulation setup."""

        args = magic_arguments.parse_argstring(self.create_setup, line)

        if not args.model:
            raise ValueError("Missing model")

        model_obj = self.shell.user_ns.get(args.model)

        if model_obj is None:
            raise KeyError(f"Model '{args.model}' not defined or not imported")
        elif not isinstance(model_obj, Model):
            raise TypeError(f"'{args.model}' is not a xsimlab.Model object")

        rendered = setup_template.format(
            model=args.model,
            in_vars=format_input_vars(
                model_obj,
                default=args.default,
                comment=args.comment,
                nested=args.nested,
            ),
        )

        content = f"# %create_setup {line}" + rendered

        self.shell.set_next_input(content, replace=True)


def load_ipython_extension(ipython):
    ipython.register_magics(SimulationMagics)
