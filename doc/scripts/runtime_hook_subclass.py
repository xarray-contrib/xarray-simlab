# TODO: use sphinx ipython directive when issue fixed
# https://github.com/ipython/ipython/issues/11362


import xsimlab as xs

import time


class PrintStepTime(xs.RuntimeHook):
    @xs.runtime_hook("run_step", "model", "pre")
    def start_step(self, model, context, state):
        self._start_time = time.time()

    @xs.runtime_hook("run_step", "model", "post")
    def finish_step(self, model, context, state):
        step_time = time.time() - self._start_time
        print(f"Step {context['step']} took {step_time:.2e} seconds")
