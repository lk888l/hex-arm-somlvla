from __future__ import annotations

import pytest

from lerobot_robot_hexbase.arm_gravity import SerialRevoluteGravityModel
from lerobot_robot_hexbase.payload_gravity import (
    lumped_inertial_from_urdf_string,
    serial_chain_endpoints,
    with_tip_payload,
)


GR80_INERTIAL_URDF = """\
<robot name="gr80">
  <link name="base_link"><inertial><origin xyz="0 0 0.04"/><mass value="0.30"/></inertial></link>
  <link name="jaw_link1"><inertial><origin xyz="0 0 0.04"/><mass value="0.05"/></inertial></link>
  <link name="jaw_link2"><inertial><origin xyz="0 0 0.04"/><mass value="0.05"/></inertial></link>
  <link name="gear_link"><inertial><origin xyz="0 0 0"/><mass value="0.01"/></inertial></link>
  <link name="tcp"/>
  <joint name="jaw_joint1" type="revolute">
    <parent link="base_link"/><child link="jaw_link1"/>
    <origin xyz="-0.01525 -0.0149 0.07679888" rpy="1.57079633 -0.97079633 1.57079633"/>
  </joint>
  <joint name="jaw_joint2" type="revolute">
    <parent link="base_link"/><child link="jaw_link2"/>
    <origin xyz="-0.01525 0.0151 0.07679888" rpy="-1.57079633 -0.97079633 -1.57079633"/>
  </joint>
  <joint name="gear_joint" type="revolute">
    <parent link="base_link"/><child link="gear_link"/>
    <origin xyz="-0.02045 0.0001 0.0365" rpy="0.78539816 -1.57079633 2.35619449"/>
  </joint>
  <joint name="tcp_joint" type="fixed">
    <parent link="base_link"/><child link="tcp"/>
    <origin xyz="0 0 0.13"/>
  </joint>
</robot>
"""

ONE_JOINT_ARM = """\
<robot name="arm">
  <link name="base"/>
  <link name="tip"/>
  <joint name="joint_1" type="revolute">
    <parent link="base"/><child link="tip"/><axis xyz="0 1 0"/>
  </joint>
</robot>
"""


def test_gr80_lumped_inertial_matches_arm_teleop_reference() -> None:
    payload = lumped_inertial_from_urdf_string(
        GR80_INERTIAL_URDF,
        expected_root_link="base_link",
        expected_tip_link="tcp",
    )
    assert payload.mass == pytest.approx(0.41, abs=1e-6)
    assert payload.center_of_mass == pytest.approx(
        (0.005537804878, 0.000026829268, 0.048889970714),
        abs=1e-7,
    )


def test_with_tip_payload_changes_terminal_gravity_without_scaling() -> None:
    payload = lumped_inertial_from_urdf_string(
        GR80_INERTIAL_URDF,
        expected_root_link="base_link",
        expected_tip_link="tcp",
    )
    arm = SerialRevoluteGravityModel.from_urdf_string(ONE_JOINT_ARM, ["joint_1"])
    enhanced = with_tip_payload(arm, payload)
    expected = -payload.mass * 9.81 * payload.center_of_mass[0]
    assert arm.gravity_torque((0.0,)) == pytest.approx((0.0,), abs=1e-8)
    assert enhanced.gravity_torque((0.0,)) == pytest.approx((expected,), abs=2e-6)


def test_payload_parser_fails_closed_on_mount_or_topology_mismatch() -> None:
    assert serial_chain_endpoints(ONE_JOINT_ARM) == ("base", "tip")
    with pytest.raises(ValueError, match="root"):
        lumped_inertial_from_urdf_string(
            GR80_INERTIAL_URDF,
            expected_root_link="wrong",
            expected_tip_link="tcp",
        )
    branched_arm = ONE_JOINT_ARM.replace(
        "</robot>",
        '<link name="extra"/><joint name="branch" type="revolute">'
        '<parent link="base"/><child link="extra"/></joint></robot>',
    )
    with pytest.raises(ValueError, match="unbranched"):
        serial_chain_endpoints(branched_arm)


@pytest.mark.parametrize(
    ("broken", "message"),
    (
        (
            GR80_INERTIAL_URDF.replace('<mass value="0.30"/>', '<mass value="nan"/>'),
            "finite",
        ),
        (
            GR80_INERTIAL_URDF.replace(
                '<origin xyz="0 0 0.04"/>', '<origin xyz="inf 0 0.04"/>', 1
            ),
            "finite",
        ),
        (
            GR80_INERTIAL_URDF.replace(
                "</robot>",
                '<joint name="second_tcp_parent" type="fixed">'
                '<parent link="jaw_link1"/><child link="tcp"/></joint></robot>',
            ),
            "multiple parents",
        ),
    ),
)
def test_payload_parser_rejects_nonfinite_or_invalid_topology(
    broken: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        lumped_inertial_from_urdf_string(
            broken,
            expected_root_link="base_link",
            expected_tip_link="tcp",
        )
