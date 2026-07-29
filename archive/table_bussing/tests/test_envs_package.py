"""Tests that exercise the envs package re-export surface."""

from __future__ import annotations

from envs import DishwareCounts, TableBussingEnv, make_env


def test_envs_package_reexports_factory():
    assert callable(make_env)
    assert issubclass(TableBussingEnv, object)
    counts = DishwareCounts(plates=1, bowls=1)
    counts.validate()
