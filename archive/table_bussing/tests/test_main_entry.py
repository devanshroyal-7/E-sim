"""Smoke-level unit test for main.py without launching ManiSkill."""

from __future__ import annotations

import main as main_mod
from table_bussing import DishwareCounts, TableBussingConfig


def test_main_uses_make_env(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    class _Env:
        def reset(self, seed=None):
            assert seed == 0
            return {}, {}

        def close(self) -> None:
            return None

        @property
        def unwrapped(self):
            class _U:
                dishware_categories = ["Plate"]
                dishware_grasps = []

            return _U()

    def _fake_make_env(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        return _Env()

    # main.py binds make_env at import time — patch the binding main uses.
    monkeypatch.setattr(main_mod, "make_env", _fake_make_env)
    main_mod.main()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) >= 2
    assert isinstance(args[0], DishwareCounts)
    assert args[0].plates == 2 and args[0].bowls == 1
    assert isinstance(args[1], TableBussingConfig)
    assert args[1].use_coacd is False
    assert kwargs.get("render_mode") == "human"
