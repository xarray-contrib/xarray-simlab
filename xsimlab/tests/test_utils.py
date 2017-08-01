import pytest

from xsimlab import utils


class TestImportRequired(object):

    def test(self):
        err_msg = "no module"
        with pytest.raises(RuntimeError) as excinfo:
            utils.import_required('this_module_does_not_exits', err_msg)
        assert err_msg in str(excinfo.value)
