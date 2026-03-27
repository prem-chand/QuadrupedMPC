"""
Validation script: manual kinematics vs MuJoCo ground truth.

Compares Go2Kinematics against mj_jacSite / data.site_xpos / data.qfrc_bias
across 100 random joint configurations.

Usage:
    python -m go2_mpc.kinematics.validate_kinematics
    # or from project root:
    python go2_mpc/kinematics/validate_kinematics.py

Expected tolerances:
  FK position:          < 1e-4 m   (floating-point, should be ~1e-12)
  Leg Jacobian (3×3):   < 1e-4 m   (geometric vs mj_jacSite)
  Full Jacobian (3×18): < 1e-4 m
  Gravity comp (qvel=0):< 0.1 Nm   (static; MuJoCo includes Coriolis)
"""

import re
import sys
from pathlib import Path

import numpy as np
import mujoco

# Allow running as a script from the project root
# __file__ = <root>/go2_mpc/kinematics/validate_kinematics.py
# parents[0] = go2_mpc/kinematics, parents[1] = go2_mpc, parents[2] = <root>
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from go2_mpc.kinematics import Go2Kinematics

_ROBOT_XML = _ROOT / "go2_mpc" / "robot" / "go2.xml"
N_SAMPLES = 100
RNG_SEED = 42

# Joint position bounds from go2.xml (per leg: [abd, hip, knee])
_Q_LEG_LO = np.array([-1.0472, -1.5708, -2.7227])
_Q_LEG_HI = np.array([ 1.0472,  3.4907, -0.83776])

_LEG_DOF_COLS = [
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [15, 16, 17],
]


def _quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """[w, x, y, z] scalar-first → (3, 3) rotation matrix."""
    w, x, y, z = q_wxyz / np.linalg.norm(q_wxyz)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _set_random_config(model, data, rng) -> tuple:
    """Set a random kinematically valid configuration, return (p_base, R_base, q_joints)."""
    q_joints = np.concatenate([
        rng.uniform(_Q_LEG_LO, _Q_LEG_HI) for _ in range(4)
    ])
    p_base = rng.uniform(-1.0, 1.0, size=3)
    q_base = rng.standard_normal(4)
    q_base /= np.linalg.norm(q_base)

    data.qpos[0:3] = p_base
    data.qpos[3:7] = q_base
    data.qpos[7:]  = q_joints
    data.qvel[:]   = 0.0

    mujoco.mj_forward(model, data)

    R_base = _quat_to_rot(q_base)
    return p_base, R_base, q_joints


# ---------------------------------------------------------------------------
# Test 1 — Forward kinematics vs data.site_xpos
# ---------------------------------------------------------------------------

def validate_fk(model, data, kin: Go2Kinematics, rng) -> None:
    foot_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ["FL_toe", "FR_toe", "RL_toe", "RR_toe"]
    ]

    errors = []
    for _ in range(N_SAMPLES):
        p_base, R_base, q_joints = _set_random_config(model, data, rng)
        for i in range(4):
            p_manual = kin.foot_position_world(i, p_base, R_base, q_joints[3*i:3*i+3])
            p_mujoco = data.site_xpos[foot_site_ids[i]].copy()
            errors.append(np.abs(p_manual - p_mujoco).max())

    max_err = max(errors)
    mean_err = float(np.mean(errors))
    print(f"  FK position:      max={max_err:.2e} m,  mean={mean_err:.2e} m")
    assert max_err < 1e-4, f"FK position error too large: {max_err:.2e} m"


# ---------------------------------------------------------------------------
# Test 2 — Leg Jacobian (3×3) vs mj_jacSite extracted columns
# ---------------------------------------------------------------------------

def validate_leg_jacobian(model, data, kin: Go2Kinematics, rng) -> None:
    foot_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ["FL_toe", "FR_toe", "RL_toe", "RR_toe"]
    ]
    _J  = np.zeros((3, model.nv))
    _Jr = np.zeros((3, model.nv))

    errors = []
    for _ in range(N_SAMPLES):
        p_base, R_base, q_joints = _set_random_config(model, data, rng)
        for i in range(4):
            mujoco.mj_jacSite(model, data, _J, _Jr, foot_site_ids[i])
            J_mj     = _J[:, _LEG_DOF_COLS[i]].copy()
            J_manual = kin.leg_jacobian(i, p_base, R_base, q_joints[3*i:3*i+3])
            errors.append(np.abs(J_manual - J_mj).max())

    max_err = max(errors)
    mean_err = float(np.mean(errors))
    print(f"  Leg Jacobian:     max={max_err:.2e},   mean={mean_err:.2e}")
    assert max_err < 1e-4, f"Leg Jacobian error too large: {max_err:.2e}"


# ---------------------------------------------------------------------------
# Test 3 — Full Jacobian (3×18) vs mj_jacSite
# ---------------------------------------------------------------------------

def validate_full_jacobian(model, data, kin: Go2Kinematics, rng) -> None:
    foot_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ["FL_toe", "FR_toe", "RL_toe", "RR_toe"]
    ]
    _J  = np.zeros((3, model.nv))
    _Jr = np.zeros((3, model.nv))

    errors = []
    for _ in range(N_SAMPLES):
        p_base, R_base, q_joints = _set_random_config(model, data, rng)
        for i in range(4):
            mujoco.mj_jacSite(model, data, _J, _Jr, foot_site_ids[i])
            J_mj     = _J.copy()
            J_manual = kin.full_jacobian(i, p_base, R_base, q_joints[3*i:3*i+3])
            errors.append(np.abs(J_manual - J_mj).max())

    max_err = max(errors)
    mean_err = float(np.mean(errors))
    print(f"  Full Jacobian:    max={max_err:.2e},   mean={mean_err:.2e}")
    assert max_err < 1e-4, f"Full Jacobian error too large: {max_err:.2e}"


# ---------------------------------------------------------------------------
# Test 4 — Static gravity comp (qvel=0) vs qfrc_bias
# ---------------------------------------------------------------------------

def validate_gravity_comp(model, data, kin: Go2Kinematics, rng) -> None:
    """
    At qvel=0 the Coriolis term vanishes, so qfrc_bias = g(q) only.
    The manual static approximation should then match MuJoCo exactly
    (up to floating-point rounding).  Any systematic error > 0.01 Nm
    indicates a CoM constant bug (most likely RL/RR hip X-sign).
    """
    errors = []
    for _ in range(N_SAMPLES):
        p_base, R_base, q_joints = _set_random_config(model, data, rng)
        # Ensure zero velocity so Coriolis contribution is zero
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        for i in range(4):
            tau_manual = kin.gravity_compensation(i, p_base, R_base, q_joints[3*i:3*i+3])
            tau_mujoco = data.qfrc_bias[_LEG_DOF_COLS[i]].copy()
            errors.append(np.abs(tau_manual - tau_mujoco).max())

    max_err = max(errors)
    mean_err = float(np.mean(errors))
    print(f"  Gravity comp:     max={max_err:.2e} Nm, mean={mean_err:.2e} Nm")
    assert max_err < 0.1, (
        f"Static gravity comp error too large: {max_err:.2e} Nm\n"
        "  Likely cause: wrong CoM constant for RL/RR hip (check HIP_COM X-sign)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_model() -> mujoco.MjModel:
    """Load go2.xml, stripping `autolimits` for MuJoCo < 2.3 compatibility."""
    xml_text = _ROBOT_XML.read_text()
    xml_text = re.sub(r'\s+autolimits="[^"]*"', "", xml_text)
    # Resolve assets relative to the XML directory
    import os
    old_cwd = os.getcwd()
    try:
        os.chdir(_ROBOT_XML.parent)
        model = mujoco.MjModel.from_xml_string(xml_text)
    finally:
        os.chdir(old_cwd)
    return model


def main() -> None:
    model = _load_model()
    data  = mujoco.MjData(model)
    kin   = Go2Kinematics()
    rng   = np.random.default_rng(RNG_SEED)

    print(f"=== Go2 Kinematics Validation  ({N_SAMPLES} random configs) ===")
    validate_fk(model, data, kin, rng)
    validate_leg_jacobian(model, data, kin, rng)
    validate_full_jacobian(model, data, kin, rng)
    validate_gravity_comp(model, data, kin, rng)
    print("All tests passed.")


if __name__ == "__main__":
    main()
