"""
Verifies MujocoRobot's kinematics backend is now injectable without
changing its default (backward-compatible) behavior.
"""
import mujoco
import numpy as np
from unittest.mock import MagicMock

from go2_mpc.core.mujoco_robot import MujocoRobot
from go2_mpc.kinematics.go2_kinematics import Go2Kinematics


def _make_model_data():
    model = mujoco.MjModel.from_xml_path("go2_mpc/robot/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_default_constructor_still_uses_go2_kinematics():
    model, data = _make_model_data()
    robot = MujocoRobot(model, data)
    assert isinstance(robot._kin, Go2Kinematics)


def test_kinematics_backend_is_injectable():
    model, data = _make_model_data()
    fake_kin = MagicMock()
    robot = MujocoRobot(model, data, kinematics=fake_kin)
    assert robot._kin is fake_kin
