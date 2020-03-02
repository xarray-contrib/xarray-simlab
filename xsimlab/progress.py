from xsimlab.hook import RuntimeHook, runtime_hook

from tqdm.auto import tqdm


class ProgressBar(RuntimeHook):
    """
    Progress bar implementation using the tqdm package.
    For additional customization, see: https://tqdm.github.io/docs/tqdm/
    """

    def __init__(self, **kwargs):
        self.pbar_dict = {}
        if kwargs:
            self.pbar_dict.update(kwargs)

    @runtime_hook("initialize", trigger="pre")
    def init_bar(self, model, context, state):
        self.pbar_dict.update(total=context["step_total"].values)
        self.pbar_model = tqdm(**self.pbar_dict)

    @runtime_hook("run_step", trigger="post")
    def update_bar(self, mode, context, state):
        self.pbar_model.update(1)

    @runtime_hook("finalize", trigger="post")
    def close_bar(self, model, context, state):
        self.pbar_model.update(1)
        self.pbar_model.close()
