"""Strict EE-URDF payload lumping for arm gravity feed-forward."""

from __future__ import annotations
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from .arm_gravity import (
    SerialRevoluteGravityModel, _F, _IDENTITY, _Link, _M3, _V3, _ZERO,
    _add, _add_scalar, _finite_f32, _matmul, _matvec, _origin, _scale,
)


@dataclass(frozen=True, slots=True)
class LumpedInertial:
    mass: float
    center_of_mass: tuple[float, float, float]

    def __post_init__(self) -> None:
        mass = float(_finite_f32(self.mass, "payload mass"))
        if mass <= 0.0:
            raise ValueError("payload mass must be positive")
        if len(self.center_of_mass) != 3:
            raise ValueError("payload center_of_mass must contain exactly three values")
        center = tuple(
            float(_finite_f32(v, f"payload center_of_mass[{i}]"))
            for i, v in enumerate(self.center_of_mass)
        )
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "center_of_mass", center)


def _robot(xml: str, label: str) -> ET.Element:
    try:
        root = ET.fromstring(xml)
    except (ET.ParseError, TypeError) as exc:
        raise ValueError(f"invalid {label} URDF XML: {exc}") from exc
    if root.tag != "robot":
        raise ValueError(f"{label} URDF root element must be <robot>")
    return root


def serial_chain_endpoints(xml: str) -> tuple[str, str]:
    root = _robot(xml, "arm")
    links: set[str] = set()
    for element in root.findall("link"):
        name = element.get("name")
        if not name or name in links:
            raise ValueError("arm URDF link names must be non-empty and unique")
        links.add(name)
    children: set[str] = set()
    next_link: dict[str, str] = {}
    for joint in root.findall("joint"):
        p = joint.find("parent")
        c = joint.find("child")
        parent = None if p is None else p.get("link")
        child = None if c is None else c.get("link")
        if not parent or not child or parent not in links or child not in links:
            raise ValueError("arm URDF joint references an unknown or unnamed link")
        if parent in next_link or child in children:
            raise ValueError("arm URDF must be an unbranched chain")
        next_link[parent] = child
        children.add(child)
    roots = links - children
    if len(roots) != 1:
        raise ValueError("arm URDF must have exactly one root link")
    arm_root = next(iter(roots))
    tip = arm_root
    visited = {tip}
    while tip in next_link:
        tip = next_link[tip]
        if tip in visited:
            raise ValueError("arm URDF serial chain contains a cycle")
        visited.add(tip)
    if visited != links:
        raise ValueError("arm URDF is not one connected serial chain")
    return arm_root, tip


def lumped_inertial_from_urdf_string(
    xml: str, *, expected_root_link: str, expected_tip_link: str
) -> LumpedInertial:
    """Reduce all link inertials into the EE root frame at zero joint position."""
    if not expected_root_link or not expected_tip_link:
        raise ValueError("expected payload root and tip links must be non-empty")
    root = _robot(xml, "payload")
    order: list[str] = []
    inertials: dict[str, _Link] = {}
    for element in root.findall("link"):
        name = element.get("name")
        if not name or name in inertials:
            raise ValueError("payload URDF link names must be non-empty and unique")
        order.append(name)
        inertial = element.find("inertial")
        if inertial is None:
            inertials[name] = _Link(_F(0.0), _ZERO)
            continue
        mass_element = inertial.find("mass")
        if mass_element is None or mass_element.get("value") is None:
            raise ValueError(f"payload link {name!r} inertial mass is missing")
        mass = _finite_f32(mass_element.get("value"), f"payload link {name!r} mass")
        if mass < _F(0.0):
            raise ValueError(f"payload link {name!r} mass must be non-negative")
        center, _ = _origin(inertial.find("origin"), f"payload link {name!r} inertial")
        inertials[name] = _Link(mass, center)
    if expected_root_link not in inertials or expected_tip_link not in inertials:
        raise ValueError("declared payload root/tip link is absent from URDF")

    edges: dict[str, list[tuple[str, _V3, _M3]]] = {name: [] for name in inertials}
    child_links: set[str] = set()
    joint_names: set[str] = set()
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name or name in joint_names:
            raise ValueError("payload URDF joint names must be non-empty and unique")
        joint_names.add(name)
        if joint.get("type") not in ("fixed", "revolute", "continuous", "prismatic"):
            raise ValueError(f"payload joint {name!r} has unsupported type")
        p = joint.find("parent")
        c = joint.find("child")
        parent = None if p is None else p.get("link")
        child = None if c is None else c.get("link")
        if not parent or not child or parent not in inertials or child not in inertials:
            raise ValueError(f"payload joint {name!r} references an unknown link")
        if child in child_links:
            raise ValueError(f"payload link {child!r} has multiple parents")
        child_links.add(child)
        xyz, rotation = _origin(joint.find("origin"), f"payload joint {name!r} origin")
        edges[parent].append((child, xyz, rotation))
    if set(inertials) - child_links != {expected_root_link}:
        raise ValueError("payload URDF declared root does not match its unique root")
    if edges[expected_tip_link]:
        raise ValueError("payload declared tip link must be a leaf")

    transforms: dict[str, tuple[_V3, _M3]] = {
        expected_root_link: (_ZERO, _IDENTITY)
    }
    pending = [expected_root_link]
    while pending:
        parent = pending.pop()
        parent_xyz, parent_rot = transforms[parent]
        for child, xyz, rotation in edges[parent]:
            if child in transforms:
                raise ValueError("payload URDF contains a cycle")
            transforms[child] = (
                _add(parent_xyz, _matvec(parent_rot, xyz)),
                _matmul(parent_rot, rotation),
            )
            pending.append(child)
    if set(transforms) != set(inertials):
        raise ValueError("payload URDF must be one connected tree")

    mass = _F(0.0)
    moment = _ZERO
    for name in order:
        link = inertials[name]
        xyz, rotation = transforms[name]
        center = _add(xyz, _matvec(rotation, link.center_of_mass))
        mass = _add_scalar(mass, link.mass)
        moment = _add(moment, _scale(center, link.mass))
    if mass <= _F(0.0):
        raise ValueError("payload URDF total mass must be positive")
    center = tuple(float(_F(value / mass)) for value in moment)
    return LumpedInertial(float(mass), center)  # type: ignore[arg-type]


def with_tip_payload(
    model: SerialRevoluteGravityModel, payload: LumpedInertial
) -> SerialRevoluteGravityModel:
    """Rigidly add a lumped payload to the arm model terminal link."""
    payload_mass = _finite_f32(payload.mass, "payload mass")
    payload_center: _V3 = tuple(
        _finite_f32(value, f"payload center_of_mass[{i}]")
        for i, value in enumerate(payload.center_of_mass)
    )  # type: ignore[assignment]
    tip = model._links[-1]
    combined_mass = _add_scalar(tip.mass, payload_mass)
    moment = _add(
        _scale(tip.center_of_mass, tip.mass),
        _scale(payload_center, payload_mass),
    )
    center: _V3 = tuple(_F(v / combined_mass) for v in moment)  # type: ignore[assignment]
    links = (*model._links[:-1], _Link(combined_mass, center))
    return SerialRevoluteGravityModel(model.joint_names, model._joints, links)


__all__ = [
    "LumpedInertial", "lumped_inertial_from_urdf_string",
    "serial_chain_endpoints", "with_tip_payload",
]
