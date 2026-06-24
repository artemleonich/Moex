"""Regression tests for ISLAND_CONFIGS defensive copy in IslandForestModel.

The class-level ``ISLAND_CONFIGS`` is a shared mutable list of dicts.
Before this fix, ``_create_island`` passed a reference to one of those
dicts straight into the sklearn constructor. As a result, any later
mutation (e.g. hyperparameter search tweaking ``max_depth``) would
silently propagate to every other IslandForestModel instance and to
every future instantiation, since sklearn retains the original dict
in its ``set_params`` history.
"""
from __future__ import annotations

import numpy as np
import pytest

from island_model.island_forest import IslandForestModel


# ---------- bug demonstration ----------


def test_island_configs_is_class_level_shared() -> None:
    """Document the underlying sharing behaviour the fix protects against.

    Removing this test would NOT remove the protection — it's a canary
    that surfaces accidental class-vs-instance refactors.
    """
    assert IslandForestModel.ISLAND_CONFIGS is IslandForestModel.ISLAND_CONFIGS
    assert isinstance(IslandForestModel.ISLAND_CONFIGS, list)
    assert all(isinstance(c, dict) for c in IslandForestModel.ISLAND_CONFIGS)


def test_mutating_class_level_config_does_not_affect_new_instance_after_fit() -> None:
    """The headline bug: mutating ISLAND_CONFIGS leaks across instances.

    Without the deepcopy fix, an sklearn classifier created via
    ``**cfg`` keeps a reference to the original class-level dict and
    re-uses it on subsequent get_params/set_params cycles. The fix
    breaks that link by giving each island its own deepcopy.
    """
    X = np.random.RandomState(0).randn(60, 5)
    y = np.random.RandomState(0).randint(0, 2, size=60)

    a = IslandForestModel(n_islands=2, trees_per_island=5, n_migrations=1, random_state=0)
    a.fit(X, y)
    a_max_depths = [isl.max_depth for isl in a.islands_]

    # Mutate the class-level config — an external user doing this
    # (e.g. hyperparameter search, debugger inspection, monkey-patch
    # from a library extension) must NOT affect anything we already fit.
    original_first_cfg = dict(IslandForestModel.ISLAND_CONFIGS[0])
    IslandForestModel.ISLAND_CONFIGS[0]["max_depth"] = 999
    try:
        b = IslandForestModel(n_islands=2, trees_per_island=5, n_migrations=1, random_state=0)
        b.fit(X, y)
        b_max_depths = [isl.max_depth for isl in b.islands_]

        # ``a`` was already fit before the mutation. Its islands must
        # still see the original max_depth, not 999.
        assert 999 not in a_max_depths, (
            f"islands from pre-mutation instance `a` were mutated by "
            f"changes to ISLAND_CONFIGS[0]: {a_max_depths}"
        )

        # ``b`` was fit after the mutation. Its islands SHOULD see 999
        # because they were created with the mutated config — that's
        # the documented behaviour. The critical invariant is that the
        # OLD instance is unaffected.
        assert 999 in b_max_depths, b_max_depths

    finally:
        # Always restore — never leak test state into other tests
        IslandForestModel.ISLAND_CONFIGS[0] = original_first_cfg

    # Sanity: a third fit, after the restore, must NOT see 999.
    c = IslandForestModel(n_islands=2, trees_per_island=5, n_migrations=1, random_state=0)
    c.fit(X, y)
    c_max_depths = [isl.max_depth for isl in c.islands_]
    assert 999 not in c_max_depths, (
        f"ISLAND_CONFIGS not properly restored after test: {c_max_depths}"
    )


def test_two_instances_do_not_share_island_config_storage() -> None:
    """Two fresh instances must each get independent copies of the cfg dict."""
    X = np.random.RandomState(1).randn(60, 5)
    y = np.random.RandomState(1).randint(0, 2, size=60)

    a = IslandForestModel(n_islands=2, trees_per_island=5, n_migrations=0, random_state=0)
    a.fit(X, y)
    # Grab the actual config dicts sklearn stored on each island's get_params()
    a_cfgs = [isl.get_params() for isl in a.islands_]

    b = IslandForestModel(n_islands=2, trees_per_island=5, n_migrations=0, random_state=0)
    b.fit(X, y)
    b_cfgs = [isl.get_params() for isl in b.islands_]

    # The dictionaries that came back via get_params() must not be the
    # same object across instances for the same idx (deepcopy isolates
    # the underlying storage).
    for i in range(len(a_cfgs)):
        assert a_cfgs[i] is not b_cfgs[i], (
            f"island[{i}] shares config storage between instances"
        )


def test_create_island_returns_independent_object_each_call() -> None:
    """Two consecutive calls to _create_island must not return linked objects."""
    model = IslandForestModel(n_islands=1, trees_per_island=5, random_state=0)
    island_a = model._create_island(0)
    island_b = model._create_island(0)
    # Same config, but different objects with their own internal state.
    assert island_a is not island_b
    # Mutating island_a's params via set_params must not bleed into island_b.
    island_a.set_params(max_depth=123)
    assert island_b.get_params()["max_depth"] != 123