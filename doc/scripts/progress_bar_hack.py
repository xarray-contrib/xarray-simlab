# A hack around ProgressBar (monkey patch) so that it renders
# nicely in docs
import io

import xsimlab
from xsimlab import runtime_hook
from xsimlab.monitoring import ProgressBar as _ProgressBar


class ProgressBarHack(_ProgressBar):
    """Redirects progress bar outputs to a variable, and
    only display the rendered string (last line) at the end
    the simulation.

    """

    def __init__(self, **kwargs):
        super(ProgressBarHack, self).__init__(**kwargs)

        self.pbar_output = io.StringIO()
        self.tqdm_kwargs.update({"file": self.pbar_output})

    @runtime_hook("finalize", trigger="post")
    def close_bar(self, model, context, state):
        super(ProgressBarHack, self).close_bar(model, context, state)
        print(self.pbar_output.getvalue().strip().split("\r")[-1])
