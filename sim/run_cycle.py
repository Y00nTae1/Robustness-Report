#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (S4 보정 폐루프 — 전체 1사이클)
=========================================
집기(PICK) → 이송(TRANSPORT) → 정렬 보정 폐루프(ALIGN: 측정→편차→보정) →
허용오차 판정 → 배치(PLACE) 의 전체 1사이클을 상태기계로 수행하고, 보정 전/후 오차를
로그로 남긴다.

설계/범위 확정(P1 리스크 #1·#4):
  - [리스크#1] ALIGN 측정은 고해상도(1280x960) 렌더로 수행해 px 분해능을 높이고, 그
    해상도에서 재캘리브한다. 달성 가능한 정밀도를 실측해 SRS_TOL(±0.5mm,±2°) 도달성을
    확인한다(미달 시 main 로그에 실측치를 그대로 노출 — 허위 PASS 금지).
  - [리스크#4] 부품 pose 는 mocap 이 authoritative, 팔은 home 고정(측정 occlusion 회피).
    집기/배치는 그리퍼 개폐 + carry 추상화(실제 물리 파지·정밀 IK 미수행 = 제약 "IK 단순화").

종료 규약: 측정 오차가 SRS 허용오차 진입 → 수렴 종료 / 보정 상한(10) 도달 → 상한 종료.
          ALIGN 중 검출 실패 → 사이클 ABORT(크래시 없이 사유 기록).

의존: S1 boot(load/reset/camera/renderer/actuate_gripper), S3 scene_parts(부품 mocap),
      external_module/interface(통합 perception=M1+M2), m1_pose(DetectionError).
제약: mujoco/numpy/opencv-python + 표준(json/csv/math)만, ROS2 미사용, 비ASCII 우회 유지.

실행 방법:
  set PANDA_XML=  (franka_emika_panda/scene.xml 의 경로)
  python sim/run_cycle.py
  -> 1사이클 완주 로그 + 보정 전/후 오차 + ABORT 데모 후 "S4 CYCLE: PASS".
     산출물: experiments/cycle_log.csv
"""

import os
import sys
import csv
import math

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "external_module"))

import boot              # noqa: E402  (S1)
import scene_parts       # noqa: E402  (S3)
import m1_pose           # noqa: E402  (S3)
import interface         # noqa: E402  (S2/S4 통합)
from m1_pose import DetectionError  # noqa: E402


SRS_TOL = {"xy": 5e-4, "theta": math.radians(2.0)}   # ±0.5mm, ±2°
ALIGN_RES = (1280, 960)                               # 정밀 측정용 고해상도


# ---------------------------------------------------------------------------
# 보조
# ---------------------------------------------------------------------------
def within_srs(error: dict, tol: dict = SRS_TOL) -> bool:
    """per-axis 허용오차 판정: |dx|,|dy|<xy AND |dtheta|<theta."""
    return (abs(error["dx"]) < tol["xy"] and abs(error["dy"]) < tol["xy"]
            and abs(error["dtheta"]) < tol["theta"])


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _gt_error(model, data, part_info, target_pose) -> dict:
    """ground-truth(부품 실제 pose) 대비 목표 오차."""
    gt = scene_parts.get_part_pose(model, data, part_info)
    return {"dx": target_pose["x"] - gt["x"],
            "dy": target_pose["y"] - gt["y"],
            "dtheta": _wrap(target_pose["theta"] - gt["theta"])}


def _build_request(rgb, depth, target_pose, cycle):
    return {"type": "perception_request",
            "image": interface.encode_image(rgb),
            "depth": interface.encode_depth(depth),
            "target_pose": target_pose, "cycle": cycle}


def set_gripper(model, data, opening: float, settle_steps: int = 1500) -> float:
    """그리퍼 개폐(S1 actuate_gripper 재사용). opening 0=닫힘/1=열림. 손가락 간격 반환."""
    return boot.actuate_gripper(model, data, opening, settle_steps=settle_steps)


def carry_waypoints(model, data, part_info, waypoints, steps_per_seg: int = 10) -> None:
    """부품 mocap 을 웨이포인트 경로로 이송(단순 선형 보간, dynamics 무관)."""
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        for i in range(1, steps_per_seg + 1):
            t = i / steps_per_seg
            scene_parts.set_part_pose(model, data, {
                "x": a["x"] + (b["x"] - a["x"]) * t,
                "y": a["y"] + (b["y"] - a["y"]) * t,
                "theta": a["theta"] + (b["theta"] - a["theta"]) * t,
            }, part_info)


# ---------------------------------------------------------------------------
# ALIGN: 측정 → 편차 → 보정 폐루프
# ---------------------------------------------------------------------------
def align_loop(model, data, part_info, camera, renderer, calib, target_pose,
               tol: dict = SRS_TOL, max_iters: int = 10,
               render_fn=None, apply_correction: bool = True) -> dict:
    """렌더→interface(M1+M2)→편차→보정 반복. 허용오차 또는 상한 시 종료.

    render_fn() -> rgb : ALIGN 측정 렌더 소스(기본 boot.render_rgb; S6 외란 주입용).
    apply_correction   : False 면 1회 측정 후 종료(보정 미적용=대조군).
    반환: {iters, converged, reason, before_error, after_error,
           before_gt, after_gt, history}
    검출 실패 시 DetectionError 를 호출자(run_cycle)로 전파.
    """
    history = []
    before_meas = before_gt = None
    meas = gt = None
    converged, reason = False, "max_iters"

    for it in range(max_iters):
        rgb = render_fn() if render_fn is not None else boot.render_rgb(model, data, camera, renderer)
        depth = boot.render_depth(model, data, camera, renderer)
        resp = interface.process_request(_build_request(rgb, depth, target_pose, it))
        meas = resp["error"]                       # 측정(M1) 기반 편차
        gt = _gt_error(model, data, part_info, target_pose)  # 실제 편차(참조)
        if it == 0:
            before_meas, before_gt = dict(meas), dict(gt)
        history.append({"iter": it, "measured": dict(meas), "ground_truth": dict(gt),
                        "within_srs": within_srs(meas, tol)})
        if within_srs(meas, tol):
            converged, reason = True, "within_tolerance"
            break
        if not apply_correction:                   # 대조군: 보정 미적용
            reason = "no_correction"
            break
        # 보정 적용: 부품 실제 pose 에 correction 가산(보정 방향 = target 으로)
        cur = scene_parts.get_part_pose(model, data, part_info)
        corr = resp["correction"]
        scene_parts.set_part_pose(model, data, {
            "x": cur["x"] + corr["dx"], "y": cur["y"] + corr["dy"],
            "theta": cur["theta"] + corr["dtheta"]}, part_info)

    return {"iters": len(history), "converged": converged, "reason": reason,
            "before_error": before_meas, "after_error": dict(meas),
            "before_gt": before_gt, "after_gt": dict(gt), "history": history}


# ---------------------------------------------------------------------------
# 전체 1사이클 상태기계
# ---------------------------------------------------------------------------
def run_cycle(model, data, part_info, camera, renderer, calib,
              source_pose: dict, target_pose: dict,
              placement_error: dict = None,
              tol: dict = SRS_TOL, max_iters: int = 10,
              render_fn=None, apply_correction: bool = True) -> dict:
    """IDLE→PICK→TRANSPORT→ALIGN→PLACE→DONE (검출 실패 시 ABORT).

    render_fn/apply_correction 은 align_loop 으로 전달(S6 외란·대조군). 기본=S4 동작.
    """
    if placement_error is None:
        placement_error = {"dx": 0.008, "dy": -0.006, "dtheta": math.radians(3.0)}
    states = []
    result = {"states": states, "state": None, "converged": False, "reason": None,
              "iters": 0, "before": None, "after": None,
              "before_gt": None, "after_gt": None,
              "source": source_pose, "target": target_pose}

    try:
        # IDLE: 팔 home, 부품 source 배치
        boot.reset_to_home(model, data)
        scene_parts.set_part_pose(model, data, source_pose, part_info)
        states.append("IDLE")

        # PICK: 그리퍼 닫힘(집기 추상화)
        set_gripper(model, data, 0.0)
        states.append("PICK")

        # TRANSPORT: source → 배치오차 포함 스테이징(target+placement_error)
        placed = {"x": target_pose["x"] + placement_error["dx"],
                  "y": target_pose["y"] + placement_error["dy"],
                  "theta": target_pose["theta"] + placement_error["dtheta"]}
        carry_waypoints(model, data, part_info, [source_pose, placed])
        states.append("TRANSPORT")

        # ALIGN: 보정 폐루프
        al = align_loop(model, data, part_info, camera, renderer, calib,
                        target_pose, tol, max_iters,
                        render_fn=render_fn, apply_correction=apply_correction)
        states.append("ALIGN")
        result.update({"converged": al["converged"], "reason": al["reason"],
                       "iters": al["iters"],
                       "before": al["before_error"], "after": al["after_error"],
                       "before_gt": al["before_gt"], "after_gt": al["after_gt"],
                       "history": al["history"]})

        # PLACE: 그리퍼 열림(배치)
        set_gripper(model, data, 1.0)
        states.append("PLACE")
        result["state"] = "DONE"
    except DetectionError as exc:
        result["state"] = "ABORT"
        result["reason"] = f"DetectionError: {exc}"
        states.append("ABORT")

    return result


# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------
def _fmt(err):
    if err is None:
        return ("", "", "")
    return (f"{err['dx']*1000:.3f}", f"{err['dy']*1000:.3f}",
            f"{math.degrees(err['dtheta']):.3f}")


def save_cycle_log(result: dict, out_path: str) -> str:
    """보정 전/후 오차(측정·실제)를 CSV 로 저장. 비ASCII 경로 안전(파이썬 파일IO)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    rows = [["stage", "source", "dx_mm", "dy_mm", "dtheta_deg"]]
    rows.append(["before", "measured", *_fmt(result["before"])])
    rows.append(["before", "ground_truth", *_fmt(result["before_gt"])])
    rows.append(["after", "measured", *_fmt(result["after"])])
    rows.append(["after", "ground_truth", *_fmt(result["after_gt"])])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return out_path


def _norm_mm_deg(err):
    """오차 크기 요약(위치 mm + 각도 deg 분리)."""
    pos = math.hypot(err["dx"], err["dy"]) * 1000.0
    ang = abs(math.degrees(err["dtheta"]))
    return pos, ang


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    model_path = os.environ.get(
        "PANDA_XML",
        os.path.join("mujoco_menagerie", "franka_emika_panda", "scene.xml"))

    here_ext = os.path.join(_THIS_DIR, "..", "external_module")
    marker_png = scene_parts.generate_aruco_png(os.path.join(here_ext, "aruco_marker.png"))
    model, data, info = scene_parts.build_scene_with_part(model_path, marker_png)
    boot.reset_to_home(model, data)
    camera = boot.resolve_top_camera(model)
    renderer = boot.make_renderer(model, ALIGN_RES[0], ALIGN_RES[1])  # 정밀 측정용

    try:
        set_pose = lambda p: scene_parts.set_part_pose(model, data, p, info)
        render_fn = lambda: boot.render_rgb(model, data, camera, renderer)

        # 0) 고해상도 재캘리브 + 달성 정밀도 실측(리스크#1 확인)
        calib = m1_pose.calibrate_scale(set_pose, render_fn)
        print(f"[캘리브@{ALIGN_RES[0]}x{ALIGN_RES[1]}] scale={calib['scale']*1000:.4f} mm/px")
        worst = 0.0
        for gt in [{"x": 0.30, "y": 0.45, "theta": 0.0},
                   {"x": 0.42, "y": 0.46, "theta": 0.20}]:
            set_pose(gt)
            est = m1_pose.estimate_pose(render_fn(), calib)
            e = math.hypot(est["x"] - gt["x"], est["y"] - gt["y"]) * 1000.0
            worst = max(worst, e)
        print(f"[정밀도 실측] 위치 최대오차 {worst:.3f} mm "
              f"(SRS xy=±{SRS_TOL['xy']*1000:.1f} mm → "
              f"{'도달 가능' if worst < SRS_TOL['xy']*1000 else '경계/미달'})")

        # 통합 perception 활성화(더미 → 실제 M1+M2)
        interface.configure_perception(calib)

        # 1) 실패 처리 검증: 부품 화면 밖(로봇만) → DetectionError → 사이클 ABORT
        print("[실패 처리] 로봇-only 프레임 검출 시도...")
        set_pose({"x": 5.0, "y": 5.0, "theta": 0.0})
        try:
            interface.process_request(_build_request(
                boot.render_rgb(model, data, camera, renderer),
                boot.render_depth(model, data, camera, renderer),
                {"x": 0, "y": 0, "theta": 0}, 0))
            raise AssertionError("로봇-only 인데 DetectionError 미발생(오탐)")
        except DetectionError:
            print("  로봇-only → DetectionError 정상(컨투어 오탐 차단)")

        # 2) 전체 1사이클(정상): source → target, 배치오차를 보정으로 수렴
        source = {"x": 0.20, "y": 0.45, "theta": 0.0}
        target = {"x": 0.40, "y": 0.45, "theta": 0.0}
        print("\n[사이클] source=(0.20,0.45,0) → target=(0.40,0.45,0), "
              "placement_error=(+8mm,-6mm,+3°)")
        res = run_cycle(model, data, info, camera, renderer, calib, source, target)
        print(f"  상태 전이: {' → '.join(res['states'])}")
        bpos, bang = _norm_mm_deg(res["before"]); apos, aang = _norm_mm_deg(res["after"])
        gbpos, gbang = _norm_mm_deg(res["before_gt"]); gapos, gaang = _norm_mm_deg(res["after_gt"])
        for h in res["history"]:
            mp, ma = _norm_mm_deg(h["measured"])
            print(f"    iter{h['iter']}: 측정 pos={mp:.3f}mm ang={ma:.3f}° "
                  f"within_srs={h['within_srs']}")
        print(f"  종료: state={res['state']}, reason={res['reason']}, iters={res['iters']}")
        print(f"  [보정 전→후] 측정 위치 {bpos:.3f}→{apos:.3f} mm, 각도 {bang:.3f}→{aang:.3f}°")
        print(f"  [보정 전→후] 실제 위치 {gbpos:.3f}→{gapos:.3f} mm, 각도 {gbang:.3f}→{gaang:.3f}°")

        log_path = save_cycle_log(res, os.path.join(_THIS_DIR, "..", "experiments", "cycle_log.csv"))
        print(f"  로그 저장: {log_path}")

        assert res["state"] == "DONE", "정상 사이클이 DONE 으로 끝나지 않음"
        assert res["converged"], "허용오차 수렴 실패(상한 소진)"
        assert apos < bpos and aang < bang, "보정 후 측정오차가 전보다 작지 않음"
        assert gapos < gbpos, "보정 후 실제오차가 전보다 작지 않음"

        # 3) ABORT 데모: target 을 로봇 가림 영역(y≈0)에 두어 정렬 측정 실패 유도
        print("\n[ABORT 데모] target=(0.30,0.0,0) (로봇 가림 영역)")
        res2 = run_cycle(model, data, info, camera, renderer, calib,
                         {"x": 0.20, "y": 0.45, "theta": 0.0},
                         {"x": 0.30, "y": 0.0, "theta": 0.0})
        print(f"  상태 전이: {' → '.join(res2['states'])}, reason={res2['reason']}")
        assert res2["state"] == "ABORT", "가림 영역인데 ABORT 되지 않음"
    finally:
        renderer.close()

    print("\nS4 CYCLE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
