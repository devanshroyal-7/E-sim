"""Tests for TableBussing tray/success helpers (pure tensor + light env)."""

from __future__ import annotations

import torch

from table_bussing import (
    DishwareCounts,
    TableBussingConfig,
    item_centers_on_tray,
    make_env,
)


def test_item_centers_on_tray_boundaries():
    half = (0.18, 0.22, 0.01)
    tray = torch.tensor([[0.0, 0.28, 0.01]])
    # Centered above tray top → on tray.
    on = item_centers_on_tray(
        torch.tensor([[0.0, 0.28, 0.05]]), tray, half
    )
    assert bool(on[0])

    # Below tray top → off.
    below = item_centers_on_tray(
        torch.tensor([[0.0, 0.28, 0.01]]), tray, half
    )
    assert not bool(below[0])

    # Outside xy footprint → off even if high.
    outside = item_centers_on_tray(
        torch.tensor([[0.5, 0.28, 0.2]]), tray, half
    )
    assert not bool(outside[0])

    # Exactly at half-extent boundary uses strict <, so edge is off.
    edge = item_centers_on_tray(
        torch.tensor([[0.18, 0.28, 0.05]]), tray, half
    )
    assert not bool(edge[0])


def test_evaluate_and_reward_shapes_headless():
    env = make_env(
        DishwareCounts(plates=1, bowls=1, cups=0, mugs=0),
        TableBussingConfig(use_coacd=False, acronym_seed=0),
        render_mode=None,
    )
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        info = raw.evaluate()
        b = raw.num_envs
        for key in ("success", "all_on_tray", "all_static", "frac_on_tray"):
            assert tuple(info[key].shape) == (b,)
        assert len(info["dishware_meta"]) == 2
        assert len(info["dishware_grasp_counts"]) == 2

        action = env.action_space.sample()
        reward = raw.compute_dense_reward(obs=None, action=action, info=info)
        assert tuple(reward.shape) == (b,)
        assert torch.isfinite(reward).all()
    finally:
        env.close()
