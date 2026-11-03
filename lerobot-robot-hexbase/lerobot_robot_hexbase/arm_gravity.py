"""Serial-revolute URDF gravity torque matching hex-arm-dynamics 0.1.0.

The sign convention is the controller convention ``M qdd + C qd + G(q) = tau``;
therefore the returned value can be used directly as gravity feed-forward torque.
All kinematic arithmetic is deliberately rounded to IEEE-754 float32 to match the
Rust reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np


_F = np.float32
_V3 = tuple[np.float32, np.float32, np.float32]
_M3 = tuple[_V3, _V3, _V3]
_ZERO: _V3 = (_F(0.0), _F(0.0), _F(0.0))
_IDENTITY: _M3 = (
    (_F(1.0), _F(0.0), _F(0.0)),
    (_F(0.0), _F(1.0), _F(0.0)),
    (_F(0.0), _F(0.0), _F(1.0)),
)


def _add_scalar(a: np.float32, b: np.float32) -> np.float32:
    return _F(a + b)


def _mul_scalar(a: np.float32, b: np.float32) -> np.float32:
    return _F(a * b)


def _sum3(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    return _add_scalar(_add_scalar(a, b), c)


def _dot(a: _V3, b: _V3) -> np.float32:
    return _sum3(
        _mul_scalar(a[0], b[0]),
        _mul_scalar(a[1], b[1]),
        _mul_scalar(a[2], b[2]),
    )


def _cross(a: _V3, b: _V3) -> _V3:
    return (
        _F(_mul_scalar(a[1], b[2]) - _mul_scalar(a[2], b[1])),
        _F(_mul_scalar(a[2], b[0]) - _mul_scalar(a[0], b[2])),
        _F(_mul_scalar(a[0], b[1]) - _mul_scalar(a[1], b[0])),
    )


def _add(a: _V3, b: _V3) -> _V3:
    return tuple(_add_scalar(x, y) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def _sub(a: _V3, b: _V3) -> _V3:
    return tuple(_F(x - y) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def _scale(a: _V3, scalar: np.float32) -> _V3:
    return tuple(_mul_scalar(x, scalar) for x in a)  # type: ignore[return-value]


def _matvec(matrix: _M3, vector: _V3) -> _V3:
    return tuple(
        _sum3(
            _mul_scalar(row[0], vector[0]),
            _mul_scalar(row[1], vector[1]),
            _mul_scalar(row[2], vector[2]),
        )
        for row in matrix
    )  # type: ignore[return-value]


def _matmul(a: _M3, b: _M3) -> _M3:
    return tuple(
        tuple(
            _sum3(
                _mul_scalar(a[row][0], b[0][column]),
                _mul_scalar(a[row][1], b[1][column]),
                _mul_scalar(a[row][2], b[2][column]),
            )
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _rpy_to_matrix(roll: np.float32, pitch: np.float32, yaw: np.float32) -> _M3:
    cr, sr = _F(np.cos(roll)), _F(np.sin(roll))
    cp, sp = _F(np.cos(pitch)), _F(np.sin(pitch))
    cy, sy = _F(np.cos(yaw)), _F(np.sin(yaw))
    return (
        (
            _mul_scalar(cy, cp),
            _F(_mul_scalar(_mul_scalar(cy, sp), sr) - _mul_scalar(sy, cr)),
            _F(_mul_scalar(_mul_scalar(cy, sp), cr) + _mul_scalar(sy, sr)),
        ),
        (
            _mul_scalar(sy, cp),
            _F(_mul_scalar(_mul_scalar(sy, sp), sr) + _mul_scalar(cy, cr)),
            _F(_mul_scalar(_mul_scalar(sy, sp), cr) - _mul_scalar(cy, sr)),
        ),
        (_F(-sp), _mul_scalar(cp, sr), _mul_scalar(cp, cr)),
    )


def _axis_angle_to_matrix(axis: _V3, angle: np.float32) -> _M3:
    norm_squared = _sum3(
        _mul_scalar(axis[0], axis[0]),
        _mul_scalar(axis[1], axis[1]),
        _mul_scalar(axis[2], axis[2]),
    )
    norm = _F(np.sqrt(norm_squared))
    if norm < _F(1e-9):
        return _IDENTITY
    k = tuple(_F(value / norm) for value in axis)  # type: ignore[assignment]
    sine, cosine = _F(np.sin(angle)), _F(np.cos(angle))
    one_minus_cosine = _F(_F(1.0) - cosine)

    def kv(i: int, j: int) -> np.float32:
        return _mul_scalar(_mul_scalar(k[i], k[j]), one_minus_cosine)

    return (
        (
            _add_scalar(cosine, kv(0, 0)),
            _F(kv(0, 1) - _mul_scalar(k[2], sine)),
            _F(kv(0, 2) + _mul_scalar(k[1], sine)),
        ),
        (
            _F(kv(1, 0) + _mul_scalar(k[2], sine)),
            _add_scalar(cosine, kv(1, 1)),
            _F(kv(1, 2) - _mul_scalar(k[0], sine)),
        ),
        (
            _F(kv(2, 0) - _mul_scalar(k[1], sine)),
            _F(kv(2, 1) + _mul_scalar(k[0], sine)),
            _add_scalar(cosine, kv(2, 2)),
        ),
    )


def _finite_f32(value: object, label: str) -> np.float32:
    try:
        result = _F(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite float") from exc
    if not math.isfinite(float(result)):
        raise ValueError(f"{label} must be a finite float")
    return result


def _vector3(text: str | None, label: str, *, default: _V3 | None = None) -> _V3:
    if text is None:
        if default is None:
            raise ValueError(f"{label} is required")
        return default
    fields = text.split()
    if len(fields) != 3:
        raise ValueError(f"{label} must contain exactly three floats")
    return tuple(_finite_f32(value, f"{label}[{index}]") for index, value in enumerate(fields))  # type: ignore[return-value]


def _origin(element: ET.Element | None, label: str) -> tuple[_V3, _M3]:
    if element is None:
        return _ZERO, _IDENTITY
    xyz = _vector3(element.get("xyz"), f"{label}.xyz", default=_ZERO)
    rpy = _vector3(element.get("rpy"), f"{label}.rpy", default=_ZERO)
    return xyz, _rpy_to_matrix(*rpy)


@dataclass(frozen=True, slots=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin_xyz: _V3
    origin_rotation: _M3
    axis: _V3


@dataclass(frozen=True, slots=True)
class _Link:
    mass: np.float32
    center_of_mass: _V3


@dataclass(frozen=True, slots=True)
class SerialRevoluteGravityModel:
    """A strict serial-chain gravity model with Rust-compatible float32 math."""

    joint_names: tuple[str, ...]
    _joints: tuple[_Joint, ...]
    _links: tuple[_Link, ...]

    @classmethod
    def from_urdf_xml(
        cls,
        xml: str,
        expected_joint_names: Sequence[str],
    ) -> SerialRevoluteGravityModel:
        expected = tuple(expected_joint_names)
        if not expected or any(not isinstance(name, str) or not name for name in expected):
            raise ValueError("expected_joint_names must contain non-empty strings")
        if len(set(expected)) != len(expected):
            raise ValueError("expected_joint_names must be unique")
        try:
            root = ET.fromstring(xml)
        except (ET.ParseError, TypeError) as exc:
            raise ValueError(f"invalid URDF XML: {exc}") from exc
        if root.tag != "robot":
            raise ValueError("URDF root element must be <robot>")

        links: dict[str, _Link] = {}
        for element in root.findall("link"):
            name = element.get("name")
            if not name:
                raise ValueError("every URDF link must have a non-empty name")
            if name in links:
                raise ValueError(f"duplicate URDF link {name!r}")
            inertial = element.find("inertial")
            if inertial is None:
                links[name] = _Link(_F(0.0), _ZERO)
                continue
            mass_element = inertial.find("mass")
            if mass_element is None or mass_element.get("value") is None:
                raise ValueError(f"link {name!r} inertial mass is missing")
            mass = _finite_f32(mass_element.get("value"), f"link {name!r} mass")
            if mass < _F(0.0):
                raise ValueError(f"link {name!r} mass must be non-negative")
            center_of_mass, _ = _origin(inertial.find("origin"), f"link {name!r} inertial origin")
            links[name] = _Link(mass, center_of_mass)
        if not links:
            raise ValueError("URDF must contain links")

        joints_by_parent: dict[str, _Joint] = {}
        child_links: set[str] = set()
        parsed_joint_names: set[str] = set()
        all_joints = root.findall("joint")
        for element in all_joints:
            name = element.get("name")
            if not name:
                raise ValueError("every URDF joint must have a non-empty name")
            if name in parsed_joint_names:
                raise ValueError(f"duplicate URDF joint {name!r}")
            parsed_joint_names.add(name)
            if element.get("type") != "revolute":
                raise ValueError(f"joint {name!r} must be revolute")
            parent_element, child_element = element.find("parent"), element.find("child")
            parent = None if parent_element is None else parent_element.get("link")
            child = None if child_element is None else child_element.get("link")
            if not parent or not child:
                raise ValueError(f"joint {name!r} must name parent and child links")
            if parent not in links or child not in links:
                raise ValueError(f"joint {name!r} references an unknown link")
            if parent in joints_by_parent:
                raise ValueError(f"URDF is branched at parent link {parent!r}")
            if child in child_links:
                raise ValueError(f"URDF child link {child!r} has multiple parents")
            origin_xyz, origin_rotation = _origin(element.find("origin"), f"joint {name!r} origin")
            axis = _vector3(
                None if element.find("axis") is None else element.find("axis").get("xyz"),
                f"joint {name!r} axis",
                default=(_F(1.0), _F(0.0), _F(0.0)),
            )
            if float(_dot(axis, axis)) < 1e-18:
                raise ValueError(f"joint {name!r} axis must be non-zero")
            joints_by_parent[parent] = _Joint(
                name=name,
                parent=parent,
                child=child,
                origin_xyz=origin_xyz,
                origin_rotation=origin_rotation,
                axis=axis,
            )
            child_links.add(child)

        roots = set(links) - child_links
        if len(roots) != 1:
            raise ValueError(f"URDF must have exactly one root link, found {len(roots)}")
        current = next(iter(roots))
        ordered_joints: list[_Joint] = []
        ordered_links: list[_Link] = []
        visited_links = {current}
        while current in joints_by_parent:
            joint = joints_by_parent[current]
            if joint.child in visited_links:
                raise ValueError("URDF serial chain contains a cycle")
            ordered_joints.append(joint)
            ordered_links.append(links[joint.child])
            visited_links.add(joint.child)
            current = joint.child
        if len(ordered_joints) != len(all_joints) or visited_links != set(links):
            raise ValueError("URDF must be one connected serial chain")
        actual = tuple(joint.name for joint in ordered_joints)
        if actual != expected:
            raise ValueError(f"URDF joint order {actual!r} does not match expected {expected!r}")
        return cls(actual, tuple(ordered_joints), tuple(ordered_links))

    @classmethod
    def from_urdf_string(
        cls,
        xml: str,
        expected_joint_names: Sequence[str],
    ) -> SerialRevoluteGravityModel:
        """Compatibility spelling used by the Rust crate and deployment code."""

        return cls.from_urdf_xml(xml, expected_joint_names)

    @property
    def dof(self) -> int:
        return len(self._joints)

    def end_effector_pose(
        self,
        q: Sequence[float],
        tool_offset: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
        """Return the terminal-link pose in the URDF root frame.

        ``tool_offset`` is expressed in the terminal-link frame. Firefly
        deployment uses this to include the hash-bound GR80 root-to-TCP
        offset without introducing a second kinematics implementation.
        """

        if len(q) != self.dof:
            raise ValueError(f"q must contain exactly {self.dof} values")
        if len(tool_offset) != 3:
            raise ValueError("tool_offset must contain exactly three values")
        q32 = tuple(
            _finite_f32(value, f"q[{index}]") for index, value in enumerate(q)
        )
        offset32: _V3 = tuple(
            _finite_f32(value, f"tool_offset[{index}]")
            for index, value in enumerate(tool_offset)
        )  # type: ignore[assignment]

        cumulative_rotation = _IDENTITY
        cumulative_position = _ZERO
        for joint, angle in zip(self._joints, q32, strict=True):
            joint_position = _add(
                cumulative_position,
                _matvec(cumulative_rotation, joint.origin_xyz),
            )
            after_origin = _matmul(cumulative_rotation, joint.origin_rotation)
            cumulative_rotation = _matmul(
                after_origin,
                _axis_angle_to_matrix(joint.axis, angle),
            )
            cumulative_position = joint_position

        position = _add(
            cumulative_position,
            _matvec(cumulative_rotation, offset32),
        )
        return (
            tuple(float(value) for value in position),
            tuple(tuple(float(value) for value in row) for row in cumulative_rotation),
        )

    def gravity_torque(
        self,
        q: Sequence[float],
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
    ) -> tuple[float, ...]:
        if len(q) != self.dof:
            raise ValueError(f"q must contain exactly {self.dof} values")
        if len(gravity) != 3:
            raise ValueError("gravity must contain exactly three values")
        q32 = tuple(_finite_f32(value, f"q[{index}]") for index, value in enumerate(q))
        gravity32: _V3 = tuple(
            _finite_f32(value, f"gravity[{index}]") for index, value in enumerate(gravity)
        )  # type: ignore[assignment]

        cumulative_rotation = _IDENTITY
        cumulative_position = _ZERO
        joint_world: list[tuple[_V3, _V3]] = []
        link_centers: list[_V3] = []
        for joint, link, angle in zip(self._joints, self._links, q32, strict=True):
            joint_position = _add(
                cumulative_position,
                _matvec(cumulative_rotation, joint.origin_xyz),
            )
            after_origin = _matmul(cumulative_rotation, joint.origin_rotation)
            joint_axis = _matvec(after_origin, joint.axis)
            joint_world.append((joint_axis, joint_position))
            child_rotation = _matmul(
                after_origin,
                _axis_angle_to_matrix(joint.axis, angle),
            )
            link_centers.append(_add(joint_position, _matvec(child_rotation, link.center_of_mass)))
            cumulative_rotation = child_rotation
            cumulative_position = joint_position

        result: list[float] = []
        for joint_index, (joint_axis, joint_position) in enumerate(joint_world):
            torque = _F(0.0)
            for link_index in range(joint_index, self.dof):
                lever = _sub(link_centers[link_index], joint_position)
                weight = _scale(gravity32, self._links[link_index].mass)
                torque = _add_scalar(torque, _dot(joint_axis, _cross(lever, weight)))
            result.append(float(_F(-torque)))
        return tuple(result)

    def gravity_torque_with(
        self,
        q: Sequence[float],
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
    ) -> tuple[float, ...]:
        """Compatibility spelling matching ``hex-arm-dynamics``."""

        return self.gravity_torque(q, gravity)


def from_urdf_xml(
    xml: str,
    expected_joint_names: Sequence[str],
) -> SerialRevoluteGravityModel:
    """Build a strict serial-revolute model from URDF XML."""

    return SerialRevoluteGravityModel.from_urdf_xml(xml, expected_joint_names)


ArmGravityModel = SerialRevoluteGravityModel


__all__ = ["ArmGravityModel", "SerialRevoluteGravityModel", "from_urdf_xml"]
