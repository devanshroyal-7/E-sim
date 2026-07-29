"""Plan and render a PushT-v1 episode with SimPlanner."""
import gymnasium as gym
import mani_skill.envs

from planner import SimPlanner

ENV_KWARGS = dict(
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
)
K_SUBSTEPS = 10
STEP_SIZE = 0.5


def main():
    plan_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode=None)
    plan_env.reset(seed=0)
    initial_state = plan_env.unwrapped.get_state().clone()

    planner = SimPlanner(plan_env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE)
    plan = planner.plan(max_expansions=2000, threshold_value=0.02)
    print(f"Plan length: {len(plan)}")
    print(f"Plan action indices: {plan}")
    plan_env.close()

    render_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode="human")
    render_env.reset(seed=0)
    render_planner = SimPlanner(render_env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE)
    render_planner.execute_plan(initial_state, plan)
    render_env.close()


if __name__ == "__main__":
    main()
