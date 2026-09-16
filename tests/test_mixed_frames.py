"""Mixed-type DataFrames, as an imputer returns them, fit like their typed counterparts."""
import numpy as np
import pandas as pd
import pytest

from ex_fuzzy import evolutionary_fit as evf


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    typed = pd.DataFrame({'age': rng.uniform(0.2, 80.0, 120).round(1),
                          'fare': rng.uniform(5.0, 500.0, 120).round(2),
                          'sex': rng.choice(['female', 'male'], 120)})
    y = np.where((typed['sex'] == 'female') | (typed['fare'] > 250), 'survived', 'died')
    as_objects = typed.astype(object)   # what SimpleImputer(strategy='most_frequent') hands back
    return typed, as_objects, y


def test_numeric_columns_keep_their_domain_in_object_frames(frames):
    typed, as_objects, y = frames
    model = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, n_gen=2, pop_size=6, random_state=0)
    model.fit(as_objects, y)
    age = next(variable for variable in model.lvs if variable.name == 'age')
    assert age.domain() == (typed['age'].min(), typed['age'].max())
    sex = next(variable for variable in model.lvs if variable.name == 'sex')
    assert sex.linguistic_variable_names() == ['female', 'male']


def test_object_frame_fits_exactly_like_the_typed_frame(frames):
    typed, as_objects, y = frames
    settings = dict(nRules=4, nAnts=2, n_gen=3, pop_size=8, random_state=0)
    from_typed = evf.BaseFuzzyRulesClassifier(**settings).fit(typed, y)
    from_objects = evf.BaseFuzzyRulesClassifier(**settings).fit(as_objects, y)
    assert from_typed.print_rules(return_rules=True) == from_objects.print_rules(return_rules=True)
    np.testing.assert_array_equal(from_typed.predict(typed), from_objects.predict(as_objects))
