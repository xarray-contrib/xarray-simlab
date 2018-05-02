import attr
import pytest

from xsimlab import utils
from xsimlab.tests.fixture_process import ExampleProcess


def test_variables_dict():
    assert all([isinstance(var, attr.Attribute)
                for var in utils.variables_dict(ExampleProcess).values()])

    assert 'other_attrib' not in utils.variables_dict(ExampleProcess)


def test_has_method():
    assert utils.has_method(ExampleProcess(), 'compute_od_var')
    assert not utils.has_method(ExampleProcess(), 'invalid_meth')


def test_maybe_to_list():
    assert utils.maybe_to_list([1]) == [1]
    assert utils.maybe_to_list(1) == [1]


def test_import_required():
    err_msg = "no module"
    with pytest.raises(RuntimeError) as excinfo:
        utils.import_required('this_module_does_not_exits', err_msg)
    assert err_msg in str(excinfo.value)


class TestAttrMapping(object):

    @pytest.fixture
    def attr_mapping(self):
        obj = utils.AttrMapping(mapping={'a': 1, 'b': 2})
        return obj

    def test_constructor(self):
        assert utils.AttrMapping()._mapping == {}

    def test_iter(self, attr_mapping):
        assert set(attr_mapping) == {'a', 'b'}

    def test_len(self, attr_mapping):
        assert len(attr_mapping) == 2

    def test_getitem(self, attr_mapping):
        assert attr_mapping['a'] == 1

    def test_get(self, attr_mapping):
        assert attr_mapping.get('b') == 2
        assert attr_mapping.get('c', None) is None

    def test_contains(self, attr_mapping):
        assert 'a' in attr_mapping
        assert 'c' not in attr_mapping

    def test_keys(self, attr_mapping):
        assert set(attr_mapping.keys()) == {'a', 'b'}

    def test_items(self, attr_mapping):
        assert set(attr_mapping.items()) == {('a', 1), ('b', 2)}

    def test_values(self, attr_mapping):
        assert sorted(attr_mapping.values()) == [1, 2]

    def test_eq(self, attr_mapping):
        assert attr_mapping == attr_mapping
        assert attr_mapping != {'c': 3}
        assert bool(attr_mapping.__eq__(2))   # bool(NotImplemented) == True

    def test_hash(self, attr_mapping):
        assert hash(attr_mapping) == hash(frozenset([('a', 1), ('b', 2)]))

    def test_getattr(self, attr_mapping):
        assert attr_mapping.a == 1
        with pytest.raises(AttributeError) as excinfo:
            attr_mapping.c
        assert 'object has no attribute' in str(excinfo.value)

    def test_setattr(self, attr_mapping):
        attr_mapping.b = 1
        assert attr_mapping.b == 1

    def test_setattr_initialized(self, attr_mapping):
        attr_mapping._initialized = True
        with pytest.raises(AttributeError) as excinfo:
            attr_mapping.b = 1
        assert "cannot override attribute" in str(excinfo.value)

    def test_dir(self, attr_mapping):
        assert 'a' in dir(attr_mapping)
