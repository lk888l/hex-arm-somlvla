#!/usr/bin/env bash
# 生成 protobuf Python(_pb2.py)到 lerobot_robot_hexbase/proto/。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="${ROBOT_PROTO_DIR:-$HERE/../hex-robot-proto/proto}"
OUT="$HERE/lerobot_robot_hexbase/proto"

if [[ ! -f "$PROTO_DIR/arm.proto" ]]; then
  echo "robot_api proto not found: $PROTO_DIR" >&2
  echo "Set ROBOT_PROTO_DIR=/absolute/path/to/hex-robot-proto/proto" >&2
  exit 1
fi

mkdir -p "$OUT"
touch "$OUT/__init__.py"
echo "proto: $PROTO_DIR -> $OUT"

PROTO_FILES=(
  common.proto
  controller.proto
  robot.proto
  base.proto
  arm.proto
  ee.proto
)

if python3 -c 'import grpc_tools.protoc' >/dev/null 2>&1; then
  python3 -m grpc_tools.protoc -I "$PROTO_DIR" --python_out="$OUT" "${PROTO_FILES[@]}"
else
  protoc -I "$PROTO_DIR" --python_out="$OUT" "${PROTO_FILES[@]}"
fi

echo "OK"
