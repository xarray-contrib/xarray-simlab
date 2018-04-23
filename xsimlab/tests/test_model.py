import pytest
import numpy as np
from numpy.testing import assert_array_equal

from xsimlab.model import Model
from xsimlab.tests.conftest import AddOnDemand, InitProfile


class TestModelBuilder(object):

    def test_bind_processes(self, model):
        assert model._processes['profile'].__xsimlab_model__ is model
        assert model._processes['profile'].__xsimlab_name__ == 'profile'

    @pytest.mark.parametrize('p_name,expected_store_keys,expected_od_keys', [
        ('init_profile',
         {'n_points': ('init_profile', 'n_points'),
          'position': ('init_profile', 'position'),
          'u_init': ('profile', 'u')},
         {}
         ),
        ('profile',
         {'u': ('profile', 'u'),
          'u_diffs': [('roll', 'u_diff')]},
         {'u_diffs': [('add', 'u_diff')], 'u_opp': ('profile', 'u_opp')}
         ),
        ('roll',
         {'shift': ('roll', 'shift'), 'u': ('profile', 'u'),
          'u_diff': ('roll', 'u_diff')},
         {}
         ),
        ('add',
         {'offset': ('add', 'offset')},
         {'u_diff': ('add', 'u_diff')}
         )
    ])
    def test_set_process_keys(self, alternative_model, p_name,
                              expected_store_keys, expected_od_keys):
        p_obj = alternative_model._processes[p_name]
        actual_store_keys = p_obj.__xsimlab_store_keys__
        actual_od_keys = p_obj.__xsimlab_od_keys__

        # key order is not ensured for group variables
        if isinstance(expected_store_keys, list):
            actual_store_keys = set(actual_store_keys)
            expected_store_keys = set(expected_store_keys)
        if isinstance(expected_od_keys, list):
            actual_od_keys = set(actual_od_keys)
            expected_od_keys = set(expected_od_keys)

        assert actual_store_keys == expected_store_keys
        assert actual_od_keys == expected_od_keys


class TestModel(object):

    def test_update_processes(self, model, alternative_model):
        a_model = model.update_processes({'add': AddOnDemand,
                                          'init_profile': InitProfile})
        assert a_model == alternative_model

    @pytest.mark.parametrize('p_names', ['add', ['add']])
    def test_drop_processes(self, model, simple_model, p_names):
        s_model = model.drop_processes(p_names)
        assert s_model == simple_model



# @pytest.fixture
# def model(model):
#     """Override fixture defined in conftest.py, return a model
#     with values set for some of its variables.
#     """
#     model.grid.x_size.value = 10
#     model.quantity.quantity.state = np.zeros(10)
#     model.some_process.some_param.value = 1

#     return model


# class TestModel(object):

#     def test_constructor(self, model):
#         # test invalid processes
#         with pytest.raises(TypeError):
#             Model({'not_a_class': Grid()})

#         class OtherClass(object):
#             pass

#         with pytest.raises(TypeError) as excinfo:
#             Model({'invalid_class': Process})
#         assert "is not a subclass" in str(excinfo.value)

#         with pytest.raises(TypeError) as excinfo:
#             Model({'invalid_class': OtherClass})
#         assert "is not a subclass" in str(excinfo.value)

#         # test process ordering
#         expected = ['grid', 'some_process', 'other_process', 'quantity']
#         assert list(model) == expected

#         # test dict-like vs. attribute access
#         assert model['grid'] is model.grid

#         # test cyclic process dependencies
#         class CyclicProcess(Process):
#             some_param = ForeignVariable(SomeProcess, 'some_param',
#                                          provided=True)
#             some_effect = ForeignVariable(SomeProcess, 'some_effect')

#         processes = {k: type(v) for k, v in model.items()}
#         processes.update({'cyclic': CyclicProcess})

#         with pytest.raises(ValueError) as excinfo:
#             Model(processes)
#         assert "cycle detected" in str(excinfo.value)

#     def test_input_vars(self, model):
#         expected = {'grid': ['x_size'],
#                     'some_process': ['some_param'],
#                     'other_process': ['other_param'],
#                     'quantity': ['quantity']}
#         actual = {k: list(v.keys()) for k, v in model.input_vars.items()}
#         assert expected == actual

#     def test_is_input(self, model):
#         assert model.is_input(model.grid.x_size) is True
#         assert model.is_input(('grid', 'x_size')) is True
#         assert model.is_input(model.quantity.all_effects) is False
#         assert model.is_input(('other_process', 'copy_param')) is False

#         external_variable = Variable(())
#         assert model.is_input(external_variable) is False

#         var_list = [Variable(()), Variable(()), Variable(())]
#         variable_list = VariableList(var_list)
#         assert model.is_input(variable_list) is False

#         variable_group = VariableGroup('group')
#         variable_group._set_variables({})
#         assert model.is_input(variable_group) is False

#     def test_visualize(self, model):
#         pytest.importorskip('graphviz')
#         ipydisp = pytest.importorskip('IPython.display')

#         result = model.visualize()
#         assert isinstance(result, ipydisp.Image)

#         result = model.visualize(show_inputs=True)
#         assert isinstance(result, ipydisp.Image)

#         result = model.visualize(show_variables=True)
#         assert isinstance(result, ipydisp.Image)

#         result = model.visualize(
#             show_only_variable=('quantity', 'quantity'))
#         assert isinstance(result, ipydisp.Image)

#     def test_initialize(self, model):
#         model.initialize()
#         expected = np.arange(10)
#         assert_array_equal(model.grid.x.value, expected)

#     def test_run_step(self, model):
#         model.initialize()
#         model.run_step(100)

#         expected = model.grid.x.value * 2
#         assert_array_equal(model.quantity.quantity.change, expected)

#     def test_finalize_step(self, model):
#         model.initialize()
#         model.run_step(100)
#         model.finalize_step()

#         expected = model.grid.x.value * 2
#         assert_array_equal(model.quantity.quantity.state, expected)

#     def test_finalize(self, model):
#         model.finalize()
#         assert model.some_process.some_effect.rate == 0

#     def test_clone(self, model):
#         cloned = model.clone()

#         for (ck, cp), (k, p) in zip(cloned.items(), model.items()):
#             assert ck == k
#             assert cp is not p

#     def test_update_processes(self, model):
#         expected = Model({'grid': Grid,
#                           'plug_process': PlugProcess,
#                           'some_process': SomeProcess,
#                           'other_process': OtherProcess,
#                           'quantity': Quantity})
#         actual = model.update_processes({'plug_process': PlugProcess})
#         assert list(actual) == list(expected)

#     def test_drop_processes(self, model):

#         expected = Model({'grid': Grid,
#                           'some_process': SomeProcess,
#                           'quantity': Quantity})
#         actual = model.drop_processes('other_process')
#         assert list(actual) == list(expected)

#         expected = Model({'grid': Grid,
#                           'quantity': Quantity})
#         actual = model.drop_processes(['some_process', 'other_process'])
#         assert list(actual) == list(expected)

#     def test_repr(self, model, model_repr):
#         assert repr(model) == model_repr

#         expected = "<xsimlab.Model (0 processes, 0 inputs)>"
#         assert repr(Model({})) == expected
