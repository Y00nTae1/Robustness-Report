#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (M2 정렬 산출 — S3 비전 모듈)
=======================================
M1 추정 pose 와 목표 pose 로부터 편차(error)와 보정 벡터(correction)를 산출한다.
S2 응답 계약과 호환되는 dict 를 반환한다.

규약(S2 인계 #1 — 보정 방향 확정):
  - error = target - estimated  (theta 는 wrap)
  - correction = gain * error   → '부품을 target 으로 이동'시키는 부호.
    즉 새 부품 pose = 현재 pose + correction 으로 적용하면 error 가 줄어든다(닫힌 루프).
  - within_tolerance = ‖error‖ < tolerance

순수 함수(좌표만, MuJoCo/이미지 비의존). 외부 라이브러리: 표준 math 만.
실행 방법(단독 self-test):
  python external_module/m2_align.py
  → 부호·크기·수렴(반복 적용 시 error→0) 검증 후 "S3 M2: PASS".
"""

import sys
import math


POSE_KEYS = ("x", "y", "theta")


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _error(estimated: dict, target: dict) -> dict:
    return {
        "dx": float(target["x"] - estimated["x"]),
        "dy": float(target["y"] - estimated["y"]),
        "dtheta": float(_wrap(target["theta"] - estimated["theta"])),
    }


def error_norm(error: dict) -> float:
    return math.sqrt(error["dx"] ** 2 + error["dy"] ** 2 + error["dtheta"] ** 2)


def compute_alignment(estimated_pose: dict, target_pose: dict,
                      gain: float = 1.0, tolerance: float = 1e-3) -> dict:
    """편차·보정 벡터·tolerance 판정. S2 응답 계약 호환."""
    err = _error(estimated_pose, target_pose)
    correction = {k: float(gain * err[k]) for k in ("dx", "dy", "dtheta")}
    return {
        "error": err,
        "correction": correction,
        "within_tolerance": bool(error_norm(err) < tolerance),
    }


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def main() -> int:
    target = {"x": 0.30, "y": 0.45, "theta": 0.0}

    # 1) 부호 검증: estimated 가 target 보다 작으면 error/correction 은 양수
    est = {"x": 0.20, "y": 0.40, "theta": -0.20}
    a = compute_alignment(est, target, gain=1.0)
    assert a["error"]["dx"] > 0 and a["error"]["dy"] > 0 and a["error"]["dtheta"] > 0, "부호 오류"
    assert a["correction"]["dx"] == a["error"]["dx"], "gain=1 시 correction=error 여야"
    assert not a["within_tolerance"]
    print(f"[부호] est={est} → error={ {k: round(v,3) for k,v in a['error'].items()} }, "
          f"correction 동일부호 OK")

    # 2) 수렴 검증: 현재 pose 에 correction 을 반복 적용하면 error_norm → 0
    cur = dict(est)
    prev = error_norm(a["error"])
    print(f"[수렴] start err_norm={prev:.4f}, gain=0.5")
    for cyc in range(10):
        al = compute_alignment(cur, target, gain=0.5, tolerance=1e-3)
        mag = error_norm(al["error"])
        print(f"  cyc={cyc} err_norm={mag:.5f} within_tol={al['within_tolerance']}")
        if cyc > 0:
            assert mag < prev, "error_norm 이 단조 감소하지 않음(보정 방향 오류)"
        if al["within_tolerance"]:
            break
        prev = mag
        cur = {"x": cur["x"] + al["correction"]["dx"],
               "y": cur["y"] + al["correction"]["dy"],
               "theta": _wrap(cur["theta"] + al["correction"]["dtheta"])}
    assert al["within_tolerance"], "수렴 실패"
    print(f"[검증] 반복 적용 수렴 OK → within_tolerance (err_norm={mag:.5f})")

    # 3) theta wrap 경계 (±pi 부근)
    a2 = compute_alignment({"x": 0, "y": 0, "theta": 3.0},
                           {"x": 0, "y": 0, "theta": -3.0})
    assert abs(a2["error"]["dtheta"]) < 0.6, "theta wrap 처리 오류"
    print(f"[wrap] θ 3.0→-3.0 error dtheta={a2['error']['dtheta']:.3f} (wrap OK)")

    print("S3 M2: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
