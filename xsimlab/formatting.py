"""Formatting utils and functions."""
from .utils import attr_fields_dict


def _calculate_col_width(col_items):
    max_name_length = (max((len(s) for s in col_items))
                       if col_items else 0)
    col_width = max(max_name_length, 7) + 6
    return col_width


def pretty_print(s, numchars):
    """Format the returned string so that it is numchars long,
    padding with trailing spaces or truncating with ellipses as
    necessary.
    """
    s = maybe_truncate(s, numchars)
    return s + ' ' * max(numchars - len(s), 0)


def maybe_truncate(s, maxlen=500):
    if len(s) > maxlen:
        s = s[:(maxlen - 3)] + '...'
    return s


def wrap_indent(text, start='', length=None):
    if length is None:
        length = len(start)
    indent = '\n' + ' ' * length
    return start + indent.join(x for x in text.splitlines())


def _summarize_var(name, var, col_width, marker=' '):
    max_line_length = 70

    first_col = pretty_print("  %s %s" % (marker, name), col_width)

    if isinstance(var, tuple):
        var_repr = "VariableList"
    else:
        var_repr = str(var).strip('<>').replace('xsimlab.', '')
        var_repr = maybe_truncate(var_repr, max_line_length - col_width)

    return first_col + var_repr


def _summarize_var_list(name, var, col_width):
    vars_lines = '\n'.join([_summarize_var('- ', v, col_width)
                            for v in var])
    return '\n'.join([_summarize_var(name, var, col_width), vars_lines])


def process_info(cls_or_obj):
    col_width = _calculate_col_width(cls_or_obj._variables)
    max_line_length = 70

    var_block = "Variables:\n"

    lines = []
    for name, var in cls_or_obj._variables.items():
        if isinstance(var, (tuple, list)):
            line = _summarize_var_list(name, var, col_width)
        else:
            marker = '*' if var.provided else ' '
            line = _summarize_var(name, var, col_width,
                                  marker=marker)
        lines.append(line)

    if not lines:
        var_block += "    *empty*"
    else:
        var_block += '\n'.join(lines)

    meta_block = "Meta:\n"
    meta_block += '\n'.join(
        [maybe_truncate("    %s: %s" % (k, v), max_line_length)
         for k, v in cls_or_obj._meta.items()]
    )

    return '\n'.join([var_block, meta_block])


def repr_process(process):
    process_cls = type(process)

    hdr = "<{} (xsimlab process)>".format(process_cls.__name__)

    variables = attr_fields_dict(process_cls)

    col_width = _calculate_col_width([k for k in variables])
    max_line_length = 70

    var_section = "Variables:\n"

    # TODO: if __xsimlab_name__ is set and not None,
    #       add process name in header
    # TODO: complete repr with variable list and possibly
    #       simulation stages implemented

    return hdr


def repr_model(model):
    n_processes = len(model)

    hdr = ("<xsimlab.Model ({} processes, {} inputs)>"
           .format(n_processes, len(model.input_vars)))

    if not n_processes:
        return hdr

    max_line_length = 70
    col_width = max([_calculate_col_width(var_name)
                     for var_name in model.input_vars])

    sections = []

    for p_name, p_obj in model.items():
        p_section = p_name

        p_input_vars = model.input_vars_dict.get(p_name, [])
        input_var_lines = []

        for var_name in p_input_vars:
            var = attr_fields_dict(type(p_obj))[var_name]
            rcol_items = []

            var_dims = " or ".join([str(d) for d in var.metadata['dims']])
            if var_dims != "()":
                rcol_items.append(var_dims)

            rcol_items += ["[{}]".format(var.metadata['intent'].value),
                           var.metadata['description']]

            line = pretty_print("    {} ".format(var_name), col_width)
            line += maybe_truncate(' '.join(rcol_items),
                                   max_line_length - col_width)

            input_var_lines.append(line)

        if input_var_lines:
            p_section += '\n' + '\n'.join(input_var_lines)
        sections.append(p_section)

    return hdr + '\n' + '\n'.join(sections)
