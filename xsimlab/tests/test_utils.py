import unittest

from xsimlab import utils


class TestImportRequired(unittest.TestCase):

    def test(self):
        err_msg = "no module"
        with self.assertRaisesRegex(RuntimeError, err_msg):
            utils.import_required('this_module_doesnt_exits', err_msg)
