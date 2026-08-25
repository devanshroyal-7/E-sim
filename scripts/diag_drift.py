"""One-shot battery: why does a 'successful' plan miss the goal in the viewer?

Default: load last_plan.txt (seed / step_size / K from its header).

    uv run scripts/diag_drift.py
    uv run scripts/diag_drift.py last_plan.txt
    uv run scripts/diag_drift.py --checks restore,roundtrip,trace,nudge,metric,settle
    uv run scripts/diag_drift.py --checks search --max-expansions 5000   # slow

Checks (all except `search` are cheap: they only replay the saved plan):
  restore    set_state round-trip + same (parent, action) after a detour
  roundtrip  search-style restore-each-prim vs execute_plan open-loop
  trace      per-primitive tee/tcp/qvel table; first mm-scale tee split
  nudge      identity set_state vs live continue at the first split
  metric     pseudo_render_intersection vs dense polygon overlap
  settle     append zero-action holds; does the search/exec gap shrink?
  search     re-run A* and check the returned plan reproduces the goal node
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_common import (
    add_plan_args,
    exact_intersection,
    intersection,
    landmark_err,
    obj_pose,
    rollout,
    setup,
    yaw_deg,
)

ALL_CHEAP = ("restore", "roundtrip", "trace", "nudge", "metric", "settle")


def banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check_restore(env, planner, root):
    banner("1. set_state: is the stored vector the whole physics state?")
    env.unwrapped.set_state(root)
    rt = float((env.unwrapped.get_state() - root).abs().max())
    print(f"set_state(root) round-trip max |delta| = {rt:.3e}")

    def child(state, act):
        env.unwrapped.set_state(state)
        for _ in range(planner.K):
            env.step(planner.action_primitives[act])
        return env.unwrapped.get_state().clone()

    a = child(root, 3)
    _ = child(root, 1)
    b = child(root, 3)
    _ = child(a, 6)
    c = child(root, 3)
    d1 = float((a - b).abs().max())
    d2 = float((a - c).abs().max())
    print(f"same (root, action=3) after detour-1 max |delta| = {d1:.3e}")
    print(f"same (root, action=3) after detour-2 max |delta| = {d2:.3e}")
    ok = rt < 1e-6 and d1 == 0.0 and d2 == 0.0
    print(
        "verdict: stored sim_state round-trips and is history-independent at rest.\n"
        "         that does NOT prove set_state is a no-op under contact — check 4."
        if ok
        else "verdict: set_state is leaky — search nodes are not well-defined"
    )
    return ok


def check_roundtrip(env, planner, root, plan):
    banner("2. search transition vs execute_plan open-loop")
    search = rollout(env, planner, root, plan, restore_every_prim=True)
    execute = rollout(env, planner, root, plan, restore_every_prim=False)
    # Heatmap uses planner._replay_trajectory_xy, which is the open-loop path.
    env.unwrapped.set_state(root)
    heat = planner._replay_trajectory_xy(root, plan)

    print(
        "step | act | search inter | exec inter | |dxy| mm | search tee xy          | exec tee xy"
    )
    first = None
    for i, snap_s in enumerate(search.snaps):
        snap_e = execute.snaps[i]
        dxy = 1000 * np.linalg.norm(snap_s.obj[:2] - snap_e.obj[:2])
        act = "-" if i == 0 else str(plan[i - 1])
        print(
            f"{i:4d} | {act:>3} | {snap_s.intersection:12.4f} | "
            f"{snap_e.intersection:10.4f} | {dxy:7.2f} | "
            f"{snap_s.obj[0]:7.4f},{snap_s.obj[1]:7.4f} | "
            f"{snap_e.obj[0]:7.4f},{snap_e.obj[1]:7.4f}"
        )
        if first is None and dxy > 0.5:
            first = i

    heat_end = heat[-1]
    exec_end = execute.snaps[-1].obj[:2]
    heat_match = np.allclose(heat, execute.obj[:, :2], atol=1e-9)
    print(
        f"\nheatmap _replay_trajectory_xy matches open-loop: {heat_match} "
        f"(end |dxy| vs exec {1000 * np.linalg.norm(heat_end - exec_end):.3f} mm)"
    )
    s_err = landmark_err(search.snaps[-1].obj, planner.goal_pose)
    e_err = landmark_err(execute.snaps[-1].obj, planner.goal_pose)
    print(
        f"search  final inter={search.snaps[-1].intersection:.4f}  "
        f"landmark err={100 * s_err:.1f} cm"
    )
    print(
        f"execute final inter={execute.snaps[-1].intersection:.4f}  "
        f"landmark err={100 * e_err:.1f} cm"
    )
    if first is None:
        print("verdict: open-loop reproduced the search. drift is elsewhere.")
    else:
        print(
            f"verdict: paths split at primitive {first} "
            f"(first |dxy| > 0.5 mm). the viewer is showing open-loop, "
            f"not the node the search accepted."
        )
    return search, execute, first


def check_trace(search, execute, plan):
    banner("3. per-primitive kinematics (search restore vs open-loop)")
    print(
        "step | act | tee|v| s/e (m/s)  | |qvel| s/e   | |d tcp| mm | |d yaw| deg"
    )
    for i, snap_s in enumerate(search.snaps):
        snap_e = execute.snaps[i]
        act = "-" if i == 0 else str(plan[i - 1])
        d_tcp = 1000 * np.linalg.norm(snap_s.tcp[:2] - snap_e.tcp[:2])
        d_yaw = abs(((yaw_deg(snap_s.obj) - yaw_deg(snap_e.obj) + 180) % 360) - 180)
        print(
            f"{i:4d} | {act:>3} | {snap_s.tee_speed:5.3f}/{snap_e.tee_speed:5.3f}     | "
            f"{snap_s.qvel:5.3f}/{snap_e.qvel:5.3f} | {d_tcp:8.2f} | {d_yaw:8.2f}"
        )
    moving = [i for i, s in enumerate(search.snaps) if s.tee_speed > 1e-3]
    print(
        f"search boundaries with tee still sliding: {moving or 'none'} "
        "(non-zero |v| means the stored node sits mid-contact)"
    )


def check_nudge(env, planner, root, plan, execute, search):
    banner("4. identity set_state at the first split: hidden contact state?")
    # Last boundary where search and open-loop still agree, then three
    # continuations of the next primitive from that live state: keep stepping,
    # set_state(get_state()) then step, and Tee xy + 1e-8 then step.
    split = None
    for i, (s, e) in enumerate(zip(search.snaps, execute.snaps)):
        if 1000 * np.linalg.norm(s.obj[:2] - e.obj[:2]) > 0.5:
            split = i
            break
    if split is None or split == 0:
        print("paths never split; nothing to probe.")
        return
    parent = split - 1
    act = plan[parent]
    print(f"first |dxy|>0.5 mm at primitive {split}; probing boundary {parent}, action {act}")

    env.unwrapped.set_state(root)
    for a in plan[:parent]:
        for _ in range(planner.K):
            env.step(planner.action_primitives[a])
    live = env.unwrapped.get_state().clone()

    env.unwrapped.set_state(live)
    restore_residual = float((env.unwrapped.get_state() - live).abs().max())

    def continue_from(state, perturb_xy=0.0):
        s = state.clone()
        if perturb_xy:
            s[0, 13] = s[0, 13] + perturb_xy
            s[0, 14] = s[0, 14] + perturb_xy
        env.unwrapped.set_state(s)
        for _ in range(planner.K):
            env.step(planner.action_primitives[act])
        return obj_pose(env).copy(), intersection(env)

    # True open-loop continue: re-walk to the boundary and step without restoring.
    env.unwrapped.set_state(root)
    for a in plan[:parent]:
        for _ in range(planner.K):
            env.step(planner.action_primitives[a])
    for _ in range(planner.K):
        env.step(planner.action_primitives[act])
    straight_pose, straight_i = obj_pose(env).copy(), intersection(env)

    restored_pose, restored_i = continue_from(live)
    nudged_pose, nudged_i = continue_from(live, perturb_xy=1e-8)

    def dxy(p):
        return 1000 * np.linalg.norm(p[:2] - straight_pose[:2])

    print(f"set_state(get_state()) residual at this boundary: {restore_residual:.3e}")
    print(f"  keep stepping (no restore)     inter={straight_i:.4f}")
    print(
        f"  set_state then step            inter={restored_i:.4f}  "
        f"|dxy| vs live {dxy(restored_pose):.3f} mm"
    )
    print(
        f"  tee xy +1e-8 then step         inter={nudged_i:.4f}  "
        f"|dxy| vs live {dxy(nudged_pose):.3f} mm"
    )
    r, n = dxy(restored_pose), dxy(nudged_pose)
    if r > 0.5 and abs(r - n) < 0.2 * max(r, 1e-6):
        print(
            "verdict: restore ≈ xy-nudge. set_state is a float-level kick; "
            "contact then amplifies it."
        )
    elif r > 0.5 and r > 3 * n:
        print(
            "verdict: restore hurts much more than a 1e-8 xy nudge. "
            "set_state is dropping contact-solver state that get_state does not save."
        )
    else:
        print("verdict: this boundary is not where the kick happens; try diag_substep.py.")


def check_metric(env, planner, root, plan, search):
    banner("5. is pseudo_render_intersection lying about the goal?")
    print("step | pseudo_render | exact overlap | diff")
    worst = 0.0
    for i, snap in enumerate(search.snaps):
        exact = exact_intersection(snap.obj, planner.goal_pose)
        diff = snap.intersection - exact
        worst = max(worst, abs(diff))
        print(f"{i:4d} | {snap.intersection:13.4f} | {exact:13.4f} | {diff:+.4f}")
    last = search.snaps[-1]
    print(
        f"\nworst |pseudo - exact| = {worst:.4f}. "
        f"at the search's last node, landmark err = "
        f"{100 * landmark_err(last.obj, planner.goal_pose):.1f} cm"
    )
    print(
        "verdict: metric is a 64x64 rasterizer (aliasing of ~0.05 is expected). "
        "it is not what puts the viewer T a whole body-length off the goal."
        if worst < 0.15
        else "verdict: metric disagrees with geometry by a lot — inspect the rasterizer."
    )


def check_settle(env, planner, root, plan):
    banner("6. does a settle phase at each primitive close the search/exec gap?")
    print("settle | search inter | exec inter | final |dxy| mm")
    for n in (0, 5, 10, 20, 40):
        s = rollout(env, planner, root, plan, restore_every_prim=True, settle=n)
        e = rollout(env, planner, root, plan, restore_every_prim=False, settle=n)
        dxy = 1000 * np.linalg.norm(s.snaps[-1].obj[:2] - e.snaps[-1].obj[:2])
        print(
            f"{n:6d} | {s.snaps[-1].intersection:12.4f} | "
            f"{e.snaps[-1].intersection:10.4f} | {dxy:8.2f}"
        )
    print(
        "verdict: shrinking |dxy| with more hold steps means the knife-edge is "
        "mid-motion at primitive boundaries. it does not by itself make the "
        "*existing* plan reach the goal — the search would have to re-plan with settle."
    )


def check_search(env, planner, args):
    banner("7. re-run A* and ask: does the returned plan reproduce the goal node?")
    import planner as planner_mod
    from search_recorder import SearchRecorder

    nodes = []

    class Traced(planner_mod.SearchNode):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            nodes.append(self)

    planner_mod.SearchNode = Traced
    root = env.unwrapped.get_state().clone()
    result = planner.plan(
        max_expansions=args.max_expansions,
        threshold_value=args.threshold,
        recorder=SearchRecorder(),
    )
    actions = result.actions
    print(f"returned plan ({len(actions)}): {actions}")
    matches = [n for n in nodes if n.action_history == actions]
    if not matches:
        print("verdict: no node has the returned action history. that is a search bug.")
        return
    node = max(matches, key=lambda n: n.intersection)
    walked = rollout(
        env, planner, root, actions, restore_every_prim=True, keep_state=True
    )
    last = walked.snaps[-1]
    delta = float((node.sim_state - last.sim_state).abs().max())
    print(f"node intersection {node.intersection:.4f}")
    print(f"walked intersection {last.intersection:.4f}")
    print(f"max |state delta| node vs walked {delta:.3e}")
    open_loop = rollout(env, planner, root, actions, restore_every_prim=False)
    print(
        f"open-loop intersection {open_loop.snaps[-1].intersection:.4f}  "
        f"landmark err {100 * landmark_err(open_loop.snaps[-1].obj, planner.goal_pose):.1f} cm"
    )
    print(
        "verdict: search is self-consistent; the returned actions only work "
        "under restore-each-prim. execute_plan is a different dynamical system."
        if delta == 0.0
        else "verdict: the action list does not rebuild the goal node even with restores."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plan_args(parser, need_search=True)
    parser.add_argument(
        "--checks",
        default="restore,roundtrip,trace,nudge,metric,settle",
        help="Comma-separated subset of: "
        + ",".join(ALL_CHEAP + ("search",)),
    )
    args = parser.parse_args()
    wanted = [c.strip() for c in args.checks.split(",") if c.strip()]
    if wanted == ["all"]:
        wanted = list(ALL_CHEAP)

    env, planner, root, plan, *_ = setup(args)
    search = execute = None
    try:
        if "restore" in wanted:
            check_restore(env, planner, root)
        if "roundtrip" in wanted or "trace" in wanted or "nudge" in wanted or "metric" in wanted:
            search, execute, _ = check_roundtrip(env, planner, root, plan)
        if "trace" in wanted:
            check_trace(search, execute, plan)
        if "nudge" in wanted:
            check_nudge(env, planner, root, plan, execute, search)
        if "metric" in wanted:
            check_metric(env, planner, root, plan, search)
        if "settle" in wanted:
            check_settle(env, planner, root, plan)
        if "search" in wanted:
            env.unwrapped.set_state(root)
            check_search(env, planner, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
