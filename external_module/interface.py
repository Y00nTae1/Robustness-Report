#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (외부 비전 모듈의 JSON 인터페이스 — S2 연동 배관)
==========================================================
시뮬에서 보낸 카메라 이미지(RGB+depth)와 목표 pose 요청을 받아, 좌표 추정/오차/보정
응답을 JSON 스키마로 되돌린다. 이 단계에서는 실제 비전을 구현하지 않고, 'cycle' 기반
결정적 감쇠로 더미 좌표를 반환한다(stateless).

JSON 계약(좌표 단위·기준):
  - pose = {"x":m, "y":m, "theta":rad}, 기준 = top 카메라 이미지 평면 투영 좌표.
  - 요청: {"type":"perception_request","image":<b64 PNG>,"depth":<b64 .npy>,
           "target_pose":{x,y,theta},"cycle":int}
  - 응답: {"estimated_pose":{x,y,theta},
           "error":{dx,dy,dtheta}, "correction":{dx,dy,dtheta},
           "within_tolerance":bool}
  (자세한 계약은 /docs/interface_contract.md 참조)

설계 메모:
  - 코덱은 메모리 버퍼(cv2.imencode / np.save(BytesIO))만 사용 → 비ASCII 경로 무관.
  - process_request(dict)->dict 가 핵심 로직. handle_json(str)->str 은 문자열 경계
    래퍼로, 후속 단계에서 소켓/서브프로세스로 무변경 전환할 수 있게 둔다.
  - 외부 라이브러리는 numpy, opencv-python(cv2) + 표준 라이브러리(json/base64/io/math).

실행 방법(단독 self-test):
  python external_module/interface.py
  → 코덱 round-trip + 스키마 + 'cycle↑ 시 error↓' 검증 후 "S2 INTERFACE: PASS".
"""

import io
import json
import math
import base64

import numpy as np
import cv2


POSE_KEYS = ("x", "y", "theta")
ERR_KEYS = ("dx", "dy", "dtheta")

# 더미 인지용 초기 오차(임의 값)와 사이클당 감쇠율. cycle 이 커질수록 오차가 단조 감소.
INITIAL_OFFSET = {"x": 0.05, "y": -0.03, "theta": 0.10}
DECAY = 0.6
DEFAULT_GAIN = 1.0
DEFAULT_TOLERANCE = 1e-3

# S4: 실제 비전 인지 설정(없으면 더미 경로 유지 — S2 회귀 방지)
_PERCEPTION = None  # {"calib":..., "dict_name":...}


def configure_perception(calib: dict, dict_name: str = "DICT_4X4_50") -> None:
    """process_request 가 더미 대신 실제 M1(검출)+M2(편차) 를 수행하도록 설정."""
    global _PERCEPTION
    _PERCEPTION = {"calib": calib, "dict_name": dict_name}


def reset_perception() -> None:
    """더미 경로로 복귀(S2 self-test 회귀용)."""
    global _PERCEPTION
    _PERCEPTION = None


# ---------------------------------------------------------------------------
# 이미지/Depth 코덱 (base64 ↔ ndarray, 메모리 버퍼만 사용)
# ---------------------------------------------------------------------------
def encode_image(rgb: np.ndarray) -> str:
    """uint8 (H,W,3) -> base64(PNG, 무손실)."""
    ok, buf = cv2.imencode(".png", rgb)
    if not ok:
        raise ValueError("이미지 PNG 인코딩 실패")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_image(s: str) -> np.ndarray:
    """base64(PNG) -> uint8 (H,W,3)."""
    raw = base64.b64decode(s.encode("ascii"))
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("이미지 PNG 디코딩 실패")
    return arr


def encode_depth(depth: np.ndarray) -> str:
    """float32 (H,W) -> base64(.npy 바이트, 원본 보존)."""
    bio = io.BytesIO()
    np.save(bio, depth.astype(np.float32, copy=False))
    return base64.b64encode(bio.getvalue()).decode("ascii")


def decode_depth(s: str) -> np.ndarray:
    """base64(.npy) -> float32 (H,W)."""
    raw = base64.b64decode(s.encode("ascii"))
    return np.load(io.BytesIO(raw))


# ---------------------------------------------------------------------------
# 스키마 검증
# ---------------------------------------------------------------------------
def validate_request(request: dict) -> None:
    if request.get("type") != "perception_request":
        raise ValueError(f"잘못된 type: {request.get('type')!r}")
    for k in ("image", "depth", "target_pose", "cycle"):
        if k not in request:
            raise ValueError(f"요청 키 누락: {k}")
    if not isinstance(request["cycle"], int):
        raise ValueError("cycle 은 int 여야 함")
    for k in POSE_KEYS:
        if k not in request["target_pose"]:
            raise ValueError(f"target_pose 키 누락: {k}")


def validate_response(response: dict) -> None:
    for k in ("estimated_pose", "error", "correction", "within_tolerance"):
        if k not in response:
            raise ValueError(f"응답 키 누락: {k}")
    for k in POSE_KEYS:
        if k not in response["estimated_pose"]:
            raise ValueError(f"estimated_pose 키 누락: {k}")
    for grp in ("error", "correction"):
        for k in ERR_KEYS:
            if k not in response[grp]:
                raise ValueError(f"{grp} 키 누락: {k}")
    if not isinstance(response["within_tolerance"], bool):
        raise ValueError("within_tolerance 는 bool 여야 함")


# ---------------------------------------------------------------------------
# 더미 인지 → 응답 생성
# ---------------------------------------------------------------------------
def build_response(target_pose: dict, cycle: int, gain: float = DEFAULT_GAIN,
                   tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """cycle 기반 결정적 감쇠로 더미 estimated/error/correction/within_tolerance 산출."""
    residual = {k: INITIAL_OFFSET[k] * (DECAY ** cycle) for k in POSE_KEYS}
    estimated = {k: float(target_pose[k] + residual[k]) for k in POSE_KEYS}
    error = {k: float(target_pose[k] - estimated[k]) for k in POSE_KEYS}  # = -residual
    correction = {k: float(gain * error[k]) for k in POSE_KEYS}
    mag = math.sqrt(sum(error[k] ** 2 for k in POSE_KEYS))
    return {
        "estimated_pose": estimated,
        "error": {"dx": error["x"], "dy": error["y"], "dtheta": error["theta"]},
        "correction": {"dx": correction["x"], "dy": correction["y"],
                       "dtheta": correction["theta"]},
        "within_tolerance": bool(mag < tolerance),
    }


def process_request(request: dict) -> dict:
    """요청 검증 → 이미지 디코드 → (설정 시)실제 M1+M2 / (미설정)더미 → 응답 검증."""
    validate_request(request)
    rgb = decode_image(request["image"])
    depth = decode_depth(request["depth"])
    if rgb.ndim != 3 or depth.ndim != 2:
        raise ValueError(f"이미지/Depth 형상 이상: rgb={rgb.shape}, depth={depth.shape}")

    if _PERCEPTION is not None:
        # 실제 비전: M1 검출 → M2 편차/보정. 검출 실패 시 DetectionError 전파.
        import m1_pose
        import m2_align
        est = m1_pose.estimate_pose(rgb, _PERCEPTION["calib"], _PERCEPTION["dict_name"])
        response = m2_align.compute_alignment(
            est, request["target_pose"], gain=DEFAULT_GAIN, tolerance=DEFAULT_TOLERANCE)
        response["estimated_pose"] = {k: float(est[k]) for k in ("x", "y", "theta")}
    else:
        response = build_response(request["target_pose"], int(request["cycle"]))

    validate_response(response)
    return response


def handle_json(request_str: str) -> str:
    """문자열 경계 래퍼(향후 socket/subprocess 전환용)."""
    return json.dumps(process_request(json.loads(request_str)))


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _error_norm(resp: dict) -> float:
    e = resp["error"]
    return math.sqrt(e["dx"] ** 2 + e["dy"] ** 2 + e["dtheta"] ** 2)


def main() -> int:
    # 1) 코덱 round-trip (무손실 동일성)
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    depth = rng.random((12, 16), dtype=np.float32)
    rgb2 = decode_image(encode_image(rgb))
    depth2 = decode_depth(encode_depth(depth))
    assert rgb2.shape == rgb.shape and rgb2.dtype == rgb.dtype, "RGB 형상/타입 불일치"
    assert np.array_equal(rgb2, rgb), "RGB round-trip 불일치"
    assert depth2.shape == depth.shape and depth2.dtype == np.float32, "Depth 형상/타입 불일치"
    assert np.array_equal(depth2, depth), "Depth round-trip 불일치(무손실 위배)"
    print("[코덱] RGB/Depth round-trip 무손실 OK")

    # 2) 스키마 + cycle↑ 시 error↓ (문자열 경계 handle_json 사용)
    target = {"x": 0.20, "y": 0.10, "theta": 0.00}
    img_b64, dep_b64 = encode_image(rgb), encode_depth(depth)
    prev = None
    for cycle in range(6):
        req = {"type": "perception_request", "image": img_b64, "depth": dep_b64,
               "target_pose": target, "cycle": cycle}
        resp = json.loads(handle_json(json.dumps(req)))
        validate_response(resp)
        mag = _error_norm(resp)
        print(f"[더미] cycle={cycle} error_norm={mag:.6f} "
              f"within_tol={resp['within_tolerance']}")
        if prev is not None:
            assert mag < prev, "cycle 증가 시 error 가 감소하지 않음"
        prev = mag

    # 3) 잘못된 요청 거부
    try:
        process_request({"type": "wrong"})
        raise AssertionError("잘못된 요청이 통과됨")
    except ValueError:
        pass
    print("[검증] 스키마/거부 로직 OK")

    print("S2 INTERFACE: PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
