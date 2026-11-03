from __future__ import annotations

import math

import pytest

from lerobot_robot_hexbase.arm_gravity import from_urdf_xml


FIREFLY_MINIMAL_URDF = """\
<robot name="firefly_y6">
  <link name="base_link"><inertial><origin xyz="-0.00015122 -0.00000305 0.04925993"/><mass value="0.87430597"/></inertial></link>
  <link name="link_1"><inertial><origin xyz="0.00671535 -0.00275632 0.02393862"/><mass value="0.25449622"/></inertial></link>
  <joint name="joint_1" type="revolute"><origin xyz="0 0 0.09505"/><parent link="base_link"/><child link="link_1"/><axis xyz="0 0 1"/></joint>
  <link name="link_2"><inertial><origin xyz="-0.00065196 0.00050605 0.13439379"/><mass value="1.28270343"/></inertial></link>
  <joint name="joint_2" type="revolute"><origin xyz="0.02 0 0.05"/><parent link="link_1"/><child link="link_2"/><axis xyz="0 1 0"/></joint>
  <link name="link_3"><inertial><origin xyz="-0.05668605 -0.00287518 0.15455959"/><mass value="0.93588267"/></inertial></link>
  <joint name="joint_3" type="revolute"><origin xyz="0 0 0.27"/><parent link="link_2"/><child link="link_3"/><axis xyz="0 1 0"/></joint>
  <link name="link_4"><inertial><origin xyz="-0.05184192 -0.00200161 0.04855136"/><mass value="0.55059939"/></inertial></link>
  <joint name="joint_4" type="revolute"><origin xyz="-0.065 0 0.25"/><parent link="link_3"/><child link="link_4"/><axis xyz="0 1 0"/></joint>
  <link name="link_5"><inertial><origin xyz="0.04236212 -0.00000333 0.00771096"/><mass value="0.41052864"/></inertial></link>
  <joint name="joint_5" type="revolute"><origin xyz="-0.047 0 0.065"/><parent link="link_4"/><child link="link_5"/><axis xyz="-1 0 0"/></joint>
  <link name="link_6"><inertial><origin xyz="-0.00048907 0 0.0034393"/><mass value="0.0076012"/></inertial></link>
  <joint name="joint_6" type="revolute"><origin xyz="0.047 0 0.034"/><parent link="link_5"/><child link="link_6"/><axis xyz="0 0 1"/></joint>
</robot>
"""
FIREFLY_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))


def test_matches_hex_arm_dynamics_v010_at_live_pose() -> None:
    model = from_urdf_xml(FIREFLY_MINIMAL_URDF, FIREFLY_JOINT_NAMES)
    q = (-0.1178539917, -1.4016660452, 3.0319726467, 0.3278790414, 0.0325877294, -0.0436512306)

    assert model.gravity_torque(q) == pytest.approx(
        (0.0, 2.151491165, -4.489645481, -0.633849740, 0.0004121332, 0.0000010250),
        abs=2e-5,
    )


def test_single_pendulum_sign_and_custom_gravity() -> None:
    xml = """
    <robot name="pendulum">
      <link name="base"/>
      <link name="bob"><inertial><origin xyz="0.5 0 0"/><mass value="2"/></inertial></link>
      <joint name="hinge" type="revolute">
        <parent link="base"/><child link="bob"/><axis xyz="0 1 0"/>
      </joint>
    </robot>
    """
    model = from_urdf_xml(xml, ("hinge",))

    assert model.gravity_torque((0.0,)) == pytest.approx((-9.81,), abs=1e-5)
    assert model.gravity_torque((math.pi / 2,)) == pytest.approx((0.0,), abs=1e-5)
    assert model.gravity_torque((0.0,), gravity=(0.0, 0.0, -4.0)) == pytest.approx((-4.0,), abs=1e-6)


def test_end_effector_pose_applies_terminal_frame_tool_offset() -> None:
    xml = """
    <robot name="one_axis">
      <link name="base"/>
      <link name="tip"/>
      <joint name="hinge" type="revolute">
        <origin xyz="0 0 0.5"/>
        <parent link="base"/><child link="tip"/><axis xyz="0 1 0"/>
      </joint>
    </robot>
    """
    model = from_urdf_xml(xml, ("hinge",))

    position, rotation = model.end_effector_pose(
        (0.0,), tool_offset=(0.0, 0.0, 0.13)
    )
    assert position == pytest.approx((0.0, 0.0, 0.63), abs=1e-6)
    for actual, expected in zip(
        rotation,
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        strict=True,
    ):
        assert actual == pytest.approx(expected)

    rotated_position, _ = model.end_effector_pose(
        (math.pi / 2,), tool_offset=(0.0, 0.0, 0.13)
    )
    assert rotated_position == pytest.approx((0.13, 0.0, 0.5), abs=1e-6)


def test_end_effector_pose_rejects_invalid_inputs() -> None:
    model = from_urdf_xml(FIREFLY_MINIMAL_URDF, FIREFLY_JOINT_NAMES)
    with pytest.raises(ValueError, match="exactly 6"):
        model.end_effector_pose((0.0,) * 5)
    with pytest.raises(ValueError, match="exactly three"):
        model.end_effector_pose((0.0,) * 6, tool_offset=(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        model.end_effector_pose(
            (0.0,) * 6, tool_offset=(0.0, math.nan, 0.0)
        )


@pytest.mark.parametrize(
    ("xml", "message"),
    [
        (
            FIREFLY_MINIMAL_URDF.replace('joint_3" type="revolute', 'joint_3" type="fixed'),
            "must be revolute",
        ),
        (
            FIREFLY_MINIMAL_URDF.replace('<axis xyz="0 1 0"/>', '<axis xyz="0 0 0"/>', 1),
            "axis must be non-zero",
        ),
        (
            FIREFLY_MINIMAL_URDF.replace('<parent link="link_1"/>', '<parent link="base_link"/>', 1),
            "branched",
        ),
    ],
)
def test_rejects_non_serial_or_invalid_urdf(xml: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        from_urdf_xml(xml, FIREFLY_JOINT_NAMES)


def test_rejects_joint_order_and_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="joint order"):
        from_urdf_xml(FIREFLY_MINIMAL_URDF, tuple(reversed(FIREFLY_JOINT_NAMES)))

    model = from_urdf_xml(FIREFLY_MINIMAL_URDF, FIREFLY_JOINT_NAMES)
    with pytest.raises(ValueError, match="exactly 6"):
        model.gravity_torque((0.0,) * 5)
    with pytest.raises(ValueError, match="finite"):
        model.gravity_torque((0.0, 0.0, math.nan, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        model.gravity_torque((0.0,) * 6, gravity=(0.0, 0.0, math.inf))
