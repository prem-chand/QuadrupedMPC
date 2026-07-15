# Generic URDF-Driven Kinematics Engine — Design Spec

**Date**: 2026-07-15
**Status**: Approved for planning
**Sub-project of**: "Extensive quadruped SDK" (multi-robot support, extensible MPC+WBC) — this is sub-project 1 of that larger goal. Sub-projects 2 (extensible control architecture) and 3 (closing current drift: teleop wiring, config/solver mismatches) are deliberately out of scope here and tracked separately.

## Problem

`QuadrupedMPC`'s controller and kinematics code has zero simulator dependencies by design (`controller/` and `kinematics/` only depend on the `Robot` ABC or plain numpy), but the *kinematics* side only works for one robot: `go2_kinematics.py` hand-derives Go2's forward kinematics, Jacobians, and gravity compensation from constants manually copied out of `go2.xml` (joint origins, link masses, CoM positions — down to a footgun where RL/RR hip CoM X sign had to be manually flipped because of how those bodies are mounted). This doesn't scale to the stated goal of supporting ANYmal, Spot, HyQ, etc. — each would require repeating this error-prone manual transcription.

## Goal

Build a generic, URDF-driven kinematics engine that computes FK, per-leg Jacobians, and gravity compensation for any robot with a 3-DOF-per-leg × 4-leg (12 joint) topology, driven by parsing the robot's URDF rather than hand-transcribing constants. Validate it first against Go2 (where two independent ground truths already exist), then against ANYmal (the first genuinely new robot).

## Explicit Non-Goals (scope boundaries)

- **Not replacing `go2_kinematics.py`.** It stays as the Go2 path (already validated to <1e-15 vs MuJoCo, fastest, zero risk to what currently works in `main.py`). The generic engine is additive, for every robot *other than* Go2 going forward, plus a one-time validation pass using Go2 as a ground-truth check.
- **Not full end-to-end ANYmal simulation.** Standing/walking ANYmal in `main.py` needs a tuned MJCF (actuator gains, contact params, stance pose) — that's separate future work. This sub-project's deliverable is a validated kinematics engine only.
- **Not full rigid-body dynamics.** No mass matrix `M(q)`, no Coriolis terms. Only FK + Jacobians + per-leg gravity compensation — exactly what the current `Robot` ABC contract needs, matching what `WholeBodyController`'s Jacobian-transpose approach actually consumes today.
- **Not generalizing beyond 3-DOF-per-leg × 4 legs.** `WholeBodyController`, `ConvexMPC`, and `controller_manager.py` all assume fixed-size `(12,)` torque/force arrays. Supporting robots with a different leg DOF count or leg count would ripple through those modules and is out of scope.
- **Not adopting Pinocchio or any URDF-parsing library.** Confirmed choice: a custom minimal parser (stdlib `xml.etree.ElementTree` only), consistent with the project's existing preference for transparent, dependency-free math.

## Architecture

```
go2_mpc/kinematics/
├── go2_kinematics.py          # unchanged — Go2's hand-derived path
├── contact_force_estimator.py # unchanged
├── frame_assertions.py        # unchanged
├── validate_kinematics.py     # unchanged (Go2 vs MuJoCo, existing)
└── generic/                   # NEW
    ├── urdf_parser.py         # URDF XML -> KinematicTree (links, joints, origins, axes, inertials)
    ├── kinematic_chain.py     # KinematicTree -> 4 base->foot KinematicChains (folds fixed joints)
    ├── generic_kinematics.py  # GenericKinematics — FK/Jacobian/gravity-comp, same duck-typed
    │                          # surface as Go2Kinematics
    └── robot_config.py        # GenericRobotConfig dataclass + per-robot factory functions
```

**Wiring change (the only touch to existing code)**: `MujocoRobot.__init__(self, model, data, kinematics=None)`, with `self._kin = kinematics or Go2Kinematics()`. Fully backward compatible — `main.py`'s current `MujocoRobot(model, data)` call is untouched. A future robot constructs `MujocoRobot(model, data, kinematics=GenericKinematics(some_robot_config()))`.

Kinematics computation method (confirmed): classical transform-chain + analytical per-joint Jacobian columns — chain the URDF's `<origin>` transforms from base to foot per leg (folding `fixed` joints into composite static offsets between actuated joints), then for each revolute joint *i* in the chain, Jacobian column = `[z_i × (p_foot − p_i); z_i]` in the world frame. This mirrors the same underlying math `go2_kinematics.py` already hand-computes, just driven by a parsed tree instead of hardcoded per-robot arrays. Gravity compensation uses mass/CoM read directly from each link's URDF `<inertial>` tag — eliminating the manual-transcription step that produced the RL/RR CoM sign-flip issue in `go2_kinematics.py`.

## Components

| File | Responsibility |
|---|---|
| `urdf_parser.py` | `parse_urdf(path) -> KinematicTree`. Parses `<link>` (name, `<inertial>` mass/CoM/inertia origin) and `<joint>` (name, type, parent, child, `<origin xyz rpy>`, `<axis xyz>`, `<limit>`) via stdlib `ElementTree`. Ignores `<visual>`/`<collision>`/mesh geometry entirely. |
| `kinematic_chain.py` | `build_leg_chains(tree, base_link, foot_links) -> list[KinematicChain]`. Walks each foot link up to `base_link`, reverses to base→foot order, folds `fixed` joints into composed static offsets between actuated joints. |
| `generic_kinematics.py` | `GenericKinematics(urdf_path, base_link, foot_links, leg_order=None)`. `foot_links` is an ordered list of 4 link names; `leg_order` defaults to `None`, meaning the order of `foot_links` itself is taken as `[FL, FR, RL, RR]` — matching the convention used throughout `controller_manager.py`/`config.py` (e.g. `foot_stance_offsets`). Parses once at construction, caches all 4 chains (axes, offsets, link mass/CoM). Exposes `forward_kinematics(q, base_pose)`, `jacobian(leg_idx, q) -> (3,3)`, `gravity_compensation(leg_idx, q, ...)` — matching `Go2Kinematics`'s public surface. |
| `robot_config.py` | `GenericRobotConfig(urdf_path, base_link, foot_links: list[str])` dataclass, `foot_links` given in `[FL, FR, RL, RR]` order. Per-robot instances as factory functions (e.g. `anymal_config()`), mirroring `config.py`'s `default_config()` pattern. |

## Data Flow

Unchanged from today except for one indirection: `GenericKinematics` parses and caches its chains once at construction (not per control step). Per step, `MujocoRobot.get_leg_jacobian(i)` / `get_gravity_compensation(i)` / `get_foot_positions_world()` delegate to `self._kin` exactly as they do today — `controller_manager.py`, `wbc.py`, `state_estimator.py` require zero changes since they only ever talk to the `Robot` ABC.

## Error Handling

All of the below raise at `GenericKinematics.__init__` (construction time) — never inside the per-step `jacobian()`/`forward_kinematics()` hot path, so a bad URDF/config fails at startup, not mid-simulation:

- Malformed XML → `URDFParseError` (wraps the underlying `ElementTree` exception with file context).
- Foot link missing from the tree, or not connected to `base_link` → `URDFTopologyError` naming the offending link.
- Unsupported joint type in a leg chain (`prismatic`, `planar`, `floating` — only `revolute`/`continuous`/`fixed` are handled) → `UnsupportedJointTypeError`.
- Zero-length or malformed `<axis>` → raise rather than silently defaulting.

## Testing & Validation

1. **Parser unit tests** — synthetic inline URDF strings (2–3 links, 1–2 joints), no real robot file needed. Covers origin/axis/inertial parsing, fixed-joint folding, and all error paths above.
2. **Kinematics unit tests** — a hand-computable trivial single-revolute-joint chain at a known angle, closed-form expected position/Jacobian. Deterministic, no external files.
3. **Real-robot validation, two stages**:
   - **Stage A — Go2 (confirmed)**: source Unitree's official Go2 URDF (from their `unitree_ros`/`unitree_mujoco` repos) and vendor it into the repo. Validate `GenericKinematics` two ways: against MuJoCo loading that same URDF directly, **and** against the existing hand-derived `Go2Kinematics` (already validated <1e-15 vs MuJoCo). Go2 is the one robot with two independent trusted oracles, making it the cheapest place to catch a parser/Jacobian bug before trusting the engine on a robot with only one oracle.
   - **Stage B — ANYmal**: source ANYmal's official URDF (ANYbotics' `anymal_c_simple_description` or similar), validate only against MuJoCo loading that URDF (no independent hand-derived ANYmal ground truth exists — that's expected, it's the actual point of the generic engine).
   - Tolerance: looser than Go2's existing <1e-15 same-code check (that compared MuJoCo to itself via `Go2Kinematics`, effectively); a reasonable target for two independent numerical paths (parser-driven chain math vs MuJoCo's own kinematics) is ~1e-6.
4. `validate_kinematics.py` gets a sibling script (not a rewrite) — e.g. `validate_generic_kinematics.py` — reusing its existing comparison-harness pattern, parameterized by robot config.

## Open Risk

Licensing/attribution for vendored third-party URDF + mesh assets (Unitree Go2, ANYbotics ANYmal) — confirm license terms permit inclusion before committing the files; note this in the implementation plan as a checklist item, not a blocker to design.
