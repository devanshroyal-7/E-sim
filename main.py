"""Minimal TableBussing-v1 entry point."""

from table_bussing import DishwareCounts, TableBussingConfig, make_env


def main() -> None:
    env = make_env(
        DishwareCounts(plates=2, bowls=1),
        TableBussingConfig(use_coacd=False),
        render_mode="human",
    )
    obs, info = env.reset(seed=0)
    print("categories:", env.unwrapped.dishware_categories)
    print("grasp counts:", [g.shape[0] for g in env.unwrapped.dishware_grasps])
    env.close()


if __name__ == "__main__":
    main()
