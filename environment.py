"""Demo / smoke-test entry for TableBussing-v1.

The environment class and canonical factory live in ``table_bussing``
(``TableBussingEnv``, ``make_env``, ``DishwareCounts``).
"""

from table_bussing import DishwareCounts, TableBussingConfig, make_env

__all__ = ["DishwareCounts", "TableBussingConfig", "make_env"]


if __name__ == "__main__":
    env = make_env(
        DishwareCounts(plates=1, bowls=1, cups=0),
        TableBussingConfig(use_coacd=False),  # faster smoke; True for better bowls
        render_mode="human",
    )
    obs, info = env.reset(seed=0)
    raw = env.unwrapped

    print("Observation keys:", obs.keys() if isinstance(obs, dict) else type(obs))
    print("Num dishware:", len(raw.dishware))
    print("Categories:", raw.dishware_categories)
    print(
        "Counts:",
        {
            "plates": raw.num_plates,
            "bowls": raw.num_bowls,
            "cups": raw.num_cups,
            "mugs": raw.num_mugs,
        },
    )
    for i, meta in enumerate(raw.dishware_meta):
        n_grasps = raw.dishware_grasps[i].shape[0]
        print(
            f"  [{i}] {meta['category']} hash={meta['mesh_hash'][:8]} "
            f"scale={meta['scale']:.6g} grasps={n_grasps} split={meta['split']}"
        )
        world = raw.get_world_grasps(i)
        print(f"       world grasps shape={world.shape}")

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()

    print("Eval:", {k: v for k, v in raw.evaluate().items() if k != "dishware_meta"})
    env.close()
