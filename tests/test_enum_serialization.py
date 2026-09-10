"""Serialization coverage for fuzzy-set types and objects that retain them."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "ex_fuzzy"
MODULE_ROOT = PACKAGE_ROOT / "ex_fuzzy"


def _run_python(source, python_path, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_path)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("python_path", "imports"),
    [
        (
            PACKAGE_ROOT,
            "from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, rules, temporal, utils",
        ),
        (
            MODULE_ROOT,
            "import evolutionary_fit as evf; import fuzzy_sets as fs; import rules; import temporal; import utils",
        ),
    ],
    ids=["package", "direct-modules"],
)
def test_fuzzy_objects_round_trip_in_fresh_process(tmp_path, python_path, imports):
    """The stable enum makes problems and rule models portable to a new process."""
    payload = tmp_path / "objects.pkl"
    writer = f"""
        import pickle
        import numpy as np
        {imports}

        original_enum = fs.FUZZY_SETS
        assert temporal.NEW_FUZZY_SETS is original_enum
        assert fs.FUZZY_SETS is original_enum

        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 1, 1])
        variables = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
        simple_rule = rules.RuleSimple(np.array([0, 1]), 0)
        rule_base = rules.MasterRuleBase([rules.RuleBaseT1(variables, [simple_rule])])
        problem = evf.FitRuleBase(
            X, y, nRules=2, nAnts=2, n_classes=2,
            linguistic_variables=variables, fuzzy_type=fs.FUZZY_SETS.t1,
        )
        model = evf.BaseFuzzyRulesClassifier(
            nRules=2, nAnts=2, linguistic_variables=variables,
            precomputed_rules=rule_base,
        )
        with open({str(payload)!r}, "wb") as stream:
            pickle.dump((list(fs.FUZZY_SETS), list(temporal.TMP_FUZZY_SETS),
                         problem, rule_base, model), stream)
    """
    reader = f"""
        import pickle
        {imports}

        with open({str(payload)!r}, "rb") as stream:
            members, temporal_members, problem, rule_base, model = pickle.load(stream)
        assert all(member is fs.FUZZY_SETS[member.name] for member in members)
        assert all(member is temporal.TMP_FUZZY_SETS[member.name]
                   for member in temporal_members)
        assert problem.fuzzy_type is fs.FUZZY_SETS.t1
        assert rule_base.fuzzy_type() is fs.FUZZY_SETS.t1
        assert model.fuzzy_type is fs.FUZZY_SETS.t1
    """

    _run_python(writer, python_path, tmp_path)
    _run_python(reader, python_path, tmp_path)


def test_package_and_direct_enum_members_compare_and_hash_equally(tmp_path):
    """Dual import support retains the library's value-based enum comparisons."""
    source = f"""
        import sys
        sys.path.insert(0, {str(MODULE_ROOT)!r})
        import fuzzy_sets as direct
        sys.path.insert(0, {str(ROOT)!r})
        from ex_fuzzy import fuzzy_sets as packaged

        for name in ("t1", "t2", "gt2", "temporal", "temporal_t2", "temporal_gt2"):
            left = direct.FUZZY_SETS[name]
            right = packaged.FUZZY_SETS[name]
            assert left == right
            assert right == left
            assert hash(left) == hash(right)
    """
    _run_python(source, "", tmp_path)


@pytest.mark.parametrize(
    ("python_path", "imports"),
    [
        (
            PACKAGE_ROOT,
            "from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, utils",
        ),
        (
            MODULE_ROOT,
            "import evolutionary_fit as evf; import fuzzy_sets as fs; import utils",
        ),
    ],
    ids=["package", "direct-modules"],
)
def test_spawned_worker_matches_parent_evaluation(
    tmp_path, python_path, imports
):
    """A spawned optimizer worker receives the problem and evaluates identically."""
    driver = tmp_path / "spawn_evaluation.py"
    driver.write_text(textwrap.dedent(f"""
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        import numpy as np
        {imports}

        def evaluate(problem, gene):
            out = {{}}
            problem._evaluate(gene, out)
            return out["F"], problem.fuzzy_type

        if __name__ == "__main__":
            X = np.array([
                [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]
            ])
            y = np.array([0, 0, 1, 1])
            variables = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
            problem = evf.FitRuleBase(
                X, y, nRules=2, nAnts=2, n_classes=2,
                linguistic_variables=variables, fuzzy_type=fs.FUZZY_SETS.t1,
            )
            gene = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1])
            expected = {{}}
            problem._evaluate(gene, expected)
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=1, mp_context=context
            ) as executor:
                actual, fuzzy_type = executor.submit(
                    evaluate, problem, gene
                ).result(timeout=30)
            np.testing.assert_array_equal(
                np.asarray(actual), np.asarray(expected["F"])
            )
            assert fuzzy_type is fs.FUZZY_SETS.t1
    """))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_path)
    subprocess.run(
        [sys.executable, str(driver)], cwd=tmp_path, env=env,
        check=True, capture_output=True, text=True,
    )
