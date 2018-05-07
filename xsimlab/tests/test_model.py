import pytest

import xsimlab as xs
from xsimlab.tests.fixture_model import AddOnDemand, InitProfile


class TestModelBuilder(object):

    def test_bind_processes(self, model):
        assert model._processes['profile'].__xsimlab_model__ is model
        assert model._processes['profile'].__xsimlab_name__ == 'profile'

    @pytest.mark.parametrize('p_name,expected_store_keys,expected_od_keys', [
        ('init_profile',
         {'n_points': ('init_profile', 'n_points'), 'u': ('profile', 'u')},
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
    def test_set_process_keys(self, model, p_name,
                              expected_store_keys, expected_od_keys):
        p_obj = model._processes[p_name]
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

    def test_get_all_variables(self, model):
        assert all([len(t) == 2 for t in model.all_vars])
        assert all([p_name in model for p_name, _ in model.all_vars])
        assert ('profile', 'u') in model.all_vars

    def test_get_input_variables(self, model):
        expected = {('init_profile', 'n_points'),
                    ('roll', 'shift'),
                    ('add', 'offset')}

        assert set(model.input_vars) == expected

    def test_get_process_dependencies(self, model):
        expected = {'init_profile': [],
                    'profile': ['init_profile', 'add', 'roll'],
                    'roll': ['init_profile'],
                    'add': []}

        actual = model.dependent_processes

        for p_name in expected:
            # order of dependencies is not ensured
            assert set(actual[p_name]) == set(expected[p_name])

    @pytest.mark.parametrize('p_name,dep_p_name', [
        ('profile', 'init_profile'),
        ('profile', 'add'),
        ('profile', 'roll'),
        ('roll', 'init_profile')
    ])
    def test_sort_processes(self, model, p_name, dep_p_name):
        p_ordered = list(model)
        assert p_ordered.index(p_name) > p_ordered.index(dep_p_name)

    def test_sort_processes_cycle(self, model):
        @xs.process
        class Foo(object):
            in_var = xs.variable()
            out_var = xs.variable(intent='out')

        @xs.process
        class Bar(object):
            in_foreign = xs.foreign(Foo, 'out_var')
            out_foreign = xs.foreign(Foo, 'in_var', intent='out')

        with pytest.raises(RuntimeError) as excinfo:
            xs.Model({'foo': Foo, 'bar': Bar})
        assert "Cycle detected" in str(excinfo.value)

    def test_get_stage_processes(self, model):
        expected = [model['roll'], model['profile']]
        assert model._p_run_step == expected


class TestModel(object):

    def test_constructor(self):
        with pytest.raises(TypeError) as excinfo:
            xs.Model({'init_profile': InitProfile()})
        assert "values must be classes" in str(excinfo.value)

        with pytest.raises(KeyError) as excinfo:
            xs.Model({'init_profile': InitProfile})
        assert "Process class 'Profile' missing" in str(excinfo.value)

        # test empty model
        assert len(xs.Model({})) == 0

    def test_process_dict_vs_attr_access(self, model):
        assert model['profile'] is model.profile

    def test_all_vars_dict(self, model):
        assert all([p_name in model for p_name in model.all_vars_dict])
        assert all([isinstance(p_vars, list)
                    for p_vars in model.all_vars_dict.values()])
        assert 'u' in model.all_vars_dict['profile']

    def test_input_vars_dict(self, model):
        assert all([p_name in model for p_name in model.input_vars_dict])
        assert all([isinstance(p_vars, list)
                    for p_vars in model.input_vars_dict.values()])
        assert 'n_points' in model.input_vars_dict['init_profile']

    def test_clone(self, model):
        cloned = model.clone()

        for p_name in model:
            assert cloned[p_name] is not model[p_name]

    def test_update_processes(self, no_init_model, model):
        m = no_init_model.update_processes({'add': AddOnDemand,
                                            'init_profile': InitProfile})
        assert m == model

    @pytest.mark.parametrize('p_names', ['add', ['add']])
    def test_drop_processes(self, no_init_model, simple_model, p_names):
        m = no_init_model.drop_processes(p_names)
        assert m == simple_model

    def test_visualize(self, model):
        pytest.importorskip('graphviz')
        ipydisp = pytest.importorskip('IPython.display')

        result = model.visualize()
        assert isinstance(result, ipydisp.Image)

        result = model.visualize(show_inputs=True)
        assert isinstance(result, ipydisp.Image)

        result = model.visualize(show_variables=True)
        assert isinstance(result, ipydisp.Image)

        result = model.visualize(
            show_only_variable=('profile', 'u'))
        assert isinstance(result, ipydisp.Image)

    def test_repr(self, simple_model, simple_model_repr):
        assert repr(simple_model) == simple_model_repr
