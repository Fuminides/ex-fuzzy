"""
Tests for the rule_mining module.

Tests rule discovery using support-based itemset mining,
rule generation, and confidence/lift pruning.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
import rules as rl
import rule_mining as rm
import utils


class TestRuleSearch:
    """Tests for the rule_search function."""

    @pytest.fixture
    def simple_data_with_variables(self):
        """Create simple data with fuzzy variables."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 3), columns=['A', 'B', 'C'])
        fuzzy_vars = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
        return X, fuzzy_vars

    def test_rule_search_finds_itemsets(self, simple_data_with_variables):
        """Test that rule_search finds frequent itemsets."""
        X, fuzzy_vars = simple_data_with_variables

        itemsets = rm.rule_search(X, fuzzy_vars, support_threshold=0.05, max_depth=2)

        assert isinstance(itemsets, list)
        assert len(itemsets) > 0

    def test_rule_search_support_threshold(self, simple_data_with_variables):
        """Test that higher threshold produces fewer itemsets."""
        X, fuzzy_vars = simple_data_with_variables

        itemsets_low = rm.rule_search(X, fuzzy_vars, support_threshold=0.01, max_depth=2)
        itemsets_high = rm.rule_search(X, fuzzy_vars, support_threshold=0.3, max_depth=2)

        assert len(itemsets_low) >= len(itemsets_high)

    def test_rule_search_max_depth(self, simple_data_with_variables):
        """Test that max_depth limits itemset size."""
        X, fuzzy_vars = simple_data_with_variables

        itemsets_depth1 = rm.rule_search(X, fuzzy_vars, support_threshold=0.05, max_depth=1)
        itemsets_depth3 = rm.rule_search(X, fuzzy_vars, support_threshold=0.05, max_depth=3)

        # With higher depth, we should find at least as many itemsets
        assert len(itemsets_depth3) >= len(itemsets_depth1)


class TestGenerateRulesFromItemsets:
    """Tests for generating rules from itemsets."""

    def test_generate_rules_basic(self):
        """Test basic rule generation from itemsets."""
        # Create sample itemsets: (antecedent_index, linguistic_variable_index)
        itemsets = [
            ((0, 0), (1, 1)),  # Var0 is LV0 AND Var1 is LV1
            ((0, 1),),        # Var0 is LV1
            ((1, 0), (2, 2)), # Var1 is LV0 AND Var2 is LV2
        ]

        rules = rm.generate_rules_from_itemsets(itemsets, nAnts=4)

        assert len(rules) == 3
        assert all(isinstance(r, rl.RuleSimple) for r in rules)

    def test_generated_rules_have_correct_structure(self):
        """Test that generated rules have correct antecedent structure."""
        itemsets = [
            ((0, 0), (1, 1)),  # Var0 is LV0 AND Var1 is LV1
        ]

        rules = rm.generate_rules_from_itemsets(itemsets, nAnts=4)

        assert len(rules) == 1
        rule = rules[0]
        assert rule.antecedents[0] == 0  # Var0 uses LV0
        assert rule.antecedents[1] == 1  # Var1 uses LV1
        assert rule.antecedents[2] == -1  # Var2 not used
        assert rule.antecedents[3] == -1  # Var3 not used


class TestMineRulebaseSupport:
    """Tests for mine_rulebase_support function."""

    @pytest.fixture
    def classification_data(self):
        """Create classification data with fuzzy variables."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])
        fuzzy_vars = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
        return X, fuzzy_vars

    def test_mine_rulebase_returns_rulebase(self, classification_data):
        """Test that mine_rulebase_support returns a RuleBase."""
        X, fuzzy_vars = classification_data

        rulebase = rm.mine_rulebase_support(
            X, fuzzy_vars,
            support_threshold=0.1,
            max_depth=2
        )

        assert isinstance(rulebase, (rl.RuleBaseT1, rl.RuleBaseT2, rl.RuleBaseGT2))

    def test_mine_rulebase_with_t2(self):
        """Test mining with Type-2 fuzzy sets."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 3), columns=['A', 'B', 'C'])
        fuzzy_vars = utils.construct_partitions(X, fs.FUZZY_SETS.t2)

        rulebase = rm.mine_rulebase_support(
            X, fuzzy_vars,
            support_threshold=0.1,
            max_depth=2
        )

        assert rulebase.fuzzy_type() == fs.FUZZY_SETS.t2


class TestMulticlassMineRulebase:
    """Tests for multiclass rule mining."""

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass classification data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(150, 4), columns=['A', 'B', 'C', 'D'])
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)
        fuzzy_vars = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
        return X, y, fuzzy_vars

    def test_multiclass_mine_returns_master_rulebase(self, multiclass_data):
        """Test that multiclass mining returns MasterRuleBase."""
        X, y, fuzzy_vars = multiclass_data

        master_rb = rm.multiclass_mine_rulebase(
            X, y, fuzzy_vars,
            support_threshold=0.1,
            max_depth=2
        )

        assert isinstance(master_rb, rl.MasterRuleBase)

    def test_multiclass_mine_has_rules_for_each_class(self, multiclass_data):
        """Test that mining produces rules for each class."""
        X, y, fuzzy_vars = multiclass_data

        master_rb = rm.multiclass_mine_rulebase(
            X, y, fuzzy_vars,
            support_threshold=0.05,
            max_depth=2,
            confidence_threshold=0.01,
            lift_threshold=0.5
        )

        # Should have rule bases for classes
        assert len(master_rb) >= 1


class TestSimpleMiningFunctions:
    """Tests for simplified mining functions."""

    def test_simple_mine_rulebase(self):
        """Test simple_mine_rulebase convenience function."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 3), columns=['A', 'B', 'C'])

        rulebase = rm.simple_mine_rulebase(
            X,
            fuzzy_type=fs.FUZZY_SETS.t1,
            support_threshold=0.1,
            max_depth=2
        )

        assert isinstance(rulebase, rl.RuleBaseT1)

    def test_simple_multiclass_mine_rulebase(self):
        """Test simple_multiclass_mine_rulebase convenience function."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 3), columns=['A', 'B', 'C'])
        y = np.array([0] * 50 + [1] * 50)

        master_rb = rm.simple_multiclass_mine_rulebase(
            X, y,
            fuzzy_type=fs.FUZZY_SETS.t1,
            support_threshold=0.1,
            max_depth=2
        )

        assert isinstance(master_rb, rl.MasterRuleBase)


class TestConfidenceLiftPruning:
    """Tests for confidence and lift based rule pruning."""

    @pytest.fixture
    def data_with_rules(self):
        """Create data and pre-mined rules."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 3), columns=['A', 'B', 'C'])
        y = np.array([0] * 50 + [1] * 50)
        fuzzy_vars = utils.construct_partitions(X, fs.FUZZY_SETS.t1)

        # Mine rules with low thresholds to get many rules
        master_rb = rm.multiclass_mine_rulebase(
            X, y, fuzzy_vars,
            support_threshold=0.01,
            max_depth=2,
            confidence_threshold=0.01,
            lift_threshold=0.1
        )

        return X, y, fuzzy_vars, master_rb

    def test_prune_rules_reduces_rule_count(self, data_with_rules):
        """Test that pruning can reduce rule count."""
        X, y, fuzzy_vars, master_rb = data_with_rules

        initial_count = len(master_rb.get_rules())

        # Prune with higher thresholds
        rm.prune_rules_confidence_lift(
            X, y, master_rb, fuzzy_vars,
            confidence_threshold=0.5,
            lift_threshold=1.1
        )

        final_count = len(master_rb.get_rules())

        # Should have same or fewer rules
        assert final_count <= initial_count


class TestCombinationGeneration:
    """Tests for internal combination generation."""

    def test_generate_combinations_basic(self):
        """Test basic combination generation."""
        lists = [[1, 2], [3, 4], [5, 6]]

        # Get combinations of length 2
        all_combs = list(rm._generate_combinations(lists, 2))

        assert len(all_combs) > 0
        # Should have combinations from pairs of lists
        for comb_iter in all_combs:
            combs = list(comb_iter)
            for c in combs:
                assert len(c) == 2

    def test_generate_combinations_single(self):
        """Test combination generation with k=1."""
        lists = [[1, 2, 3], [4, 5]]

        all_combs = list(rm._generate_combinations(lists, 1))

        assert len(all_combs) == 2  # Two lists


class TestEdgeCases:
    """Tests for edge cases in rule mining."""

    def test_mining_with_very_high_support(self):
        """Test mining with very high support threshold (may find no rules)."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(50, 3), columns=['A', 'B', 'C'])
        fuzzy_vars = utils.construct_partitions(X, fs.FUZZY_SETS.t1)

        itemsets = rm.rule_search(X, fuzzy_vars, support_threshold=0.99, max_depth=2)

        # May or may not find rules, but shouldn't crash
        assert isinstance(itemsets, list)

    def test_mining_single_feature(self):
        """Test mining with single feature."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 1), columns=['A'])
        fuzzy_vars = utils.construct_partitions(X, fs.FUZZY_SETS.t1)

        rulebase = rm.mine_rulebase_support(
            X, fuzzy_vars,
            support_threshold=0.1,
            max_depth=1
        )

        assert len(rulebase) >= 0  # May or may not find rules


class TestIntegrationWithClassifier:
    """Integration tests with classifier."""

    def test_mined_rules_usable_by_classifier(self):
        """Test that mined rules can be used by classifier."""
        from sklearn.datasets import load_iris
        import evolutionary_fit as evf

        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        fuzzy_vars = utils.construct_partitions(X_train, fs.FUZZY_SETS.t1)

        # Mine candidate rules
        candidate_rules = rm.multiclass_mine_rulebase(
            X_train, y_train, fuzzy_vars,
            support_threshold=0.05,
            max_depth=3
        )

        # Use in classifier
        clf = evf.BaseFuzzyRulesClassifier(
            nRules=15, nAnts=4,
            verbose=False
        )
        clf.fit(X_train, y_train, n_gen=10, pop_size=20, candidate_rules=candidate_rules)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
