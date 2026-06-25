#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (M1 부품 검출 — S3 비전 모듈)
=======================================
탑다운 RGB 이미지에서 광학 부품(ArUco 마커판)의 중심 좌표·주축 각도를 검출한다.
ArUco 검출을 1차로, 실패 시 OpenCV 컨투어(minAreaRect)를 보조로 쓰고, 둘 다 실패하면
DetectionError 를 명시적으로 던진다(제약). 픽셀→평면(m) 변환은 2점 캘리브로 산출한
선형 매핑(u=u0+x/s, v=v0-y/s)을 사용한다.

채널 규약(S2 인계 #3): 입력은 RGB(boot.render_rgb 출력). 내부에서 RGB→Gray 로 명시 변환.
좌표 계약: 반환 pose={x,y,theta}, 단위 x,y=m / theta=rad, 기준=top 카메라 평면.

의존: scene_parts(부품 세터/게터, 캘리브용), sim/boot(렌더). m2_align(닫힌 루프 데모).
실행 방법(단독 self-test):
  set PANDA_XML=  (franka_emika_panda/scene.xml 의 경로)
  python external_module/m1_pose.py
  -> 캘리브 -> 알려진 pose 검출 정확도 -> 검출 실패(빈 화면) 에러 -> M1+M2 닫힌 루프
     error 감소 확인 후 "S3 VISION: PASS".
"""

import os
import sys
import math

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scene_parts  # noqa: E402


class DetectionError(Exception):
    """ArUco·컨투어 모두 부품을 검출하지 못했을 때."""


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _dictionary(dict_name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# 검출기 (1차: ArUco / 보조: 컨투어)
# ---------------------------------------------------------------------------
def detect_aruco(rgb: np.ndarray, dict_name: str = scene_parts.DEFAULT_DICT):
    """ArUco 검출. 반환 {center_px:(u,v), theta_img:rad, corners} 또는 None."""
    gray = _to_gray(rgb)
    det = cv2.aruco.ArucoDetector(_dictionary(dict_name), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None
    c = corners[0].reshape(4, 2)
    center = c.mean(axis=0)
    theta_img = math.atan2(c[1][1] - c[0][1], c[1][0] - c[0][0])  # corner0→1 방향
    return {"center_px": (float(center[0]), float(center[1])),
            "theta_img": float(theta_img), "corners": c}


def detect_contour(rgb: np.ndarray, min_area: float = 1500.0,
                   max_area: float = 60000.0, aspect_tol: float = 0.3):
    """보조 검출: 형상 게이트(면적·정사각 aspect·extent)를 통과한 후보만 반환.

    형상 게이트로 로봇 블롭/그림자 오탐을 차단한다(S3 인계 #1). 후보 없으면 None.
    """
    gray = _to_gray(rgb)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
        if w < 1 or h < 1:
            continue
        aspect = w / h
        if not (1.0 - aspect_tol <= aspect <= 1.0 + aspect_tol):  # 정사각 마커판
            continue
        rect_area = w * h
        if rect_area <= 0 or (area / rect_area) < 0.6:            # 채움(extent) 충분
            continue
        if area > best_area:                                       # 통과 후보 중 최대
            best, best_area = {"center_px": (float(cx), float(cy)),
                               "theta_img": float(math.radians(ang)), "corners": None}, area
    return best


# ---------------------------------------------------------------------------
# 픽셀 ↔ 평면(m) 변환 + 캘리브레이션
# ---------------------------------------------------------------------------
def pixel_to_plane(u: float, v: float, calib: dict):
    """선형 매핑: x=(u-u0)*s, y=-(v-v0)*s."""
    s = calib["scale"]
    return ((u - calib["u0"]) * s, -(v - calib["v0"]) * s)


def calibrate_scale(set_pose, render_fn, dict_name: str = scene_parts.DEFAULT_DICT) -> dict:
    """알려진 포즈들을 mocap 으로 옮겨 렌더·검출 → scale/origin/theta 매핑 산출.

    set_pose(pose_dict)->None : 부품을 해당 pose 로 이동(+mj_forward)
    render_fn()->rgb          : 현재 상태 상단 RGB 렌더
    """
    def detect(pose):
        set_pose(pose)
        d = detect_aruco(render_fn(), dict_name)
        if d is None:
            raise DetectionError(f"캘리브 중 검출 실패 pose={pose}")
        return d

    # 로봇이 이미지 중앙 띠(|y|<~0.3)를 점유하므로 캘리브 점은 빈 영역에서 고른다.
    A = detect({"x": 0.30, "y": 0.45, "theta": 0.0})
    B = detect({"x": 0.60, "y": 0.45, "theta": 0.0})    # x 변화
    C = detect({"x": 0.30, "y": -0.45, "theta": 0.0})   # y 변화(큰 폭으로 정밀도↑)
    uA, vA = A["center_px"]; uB, _ = B["center_px"]; _, vC = C["center_px"]

    sx = (0.60 - 0.30) / (uB - uA)            # m/px (x축): u=u0+x/s
    sy = 0.90 / (vC - vA)                      # v=v0-y/s → vC-vA=0.9/s
    scale = (abs(sx) + abs(sy)) / 2.0
    u0 = uA - 0.30 / scale
    v0 = vA + 0.45 / scale

    theta_ref = A["theta_img"]
    D = detect({"x": 0.30, "y": 0.45, "theta": 0.30})
    d_img = _wrap(D["theta_img"] - theta_ref)
    theta_sign = 1.0 if d_img > 0 else -1.0
    return {"scale": scale, "u0": u0, "v0": v0,
            "theta_ref": theta_ref, "theta_sign": theta_sign, "dict_name": dict_name}


# ---------------------------------------------------------------------------
# 통합 추정
# ---------------------------------------------------------------------------
def estimate_pose(rgb: np.ndarray, calib: dict,
                  dict_name: str = scene_parts.DEFAULT_DICT) -> dict:
    """ArUco 1차 → 컨투어 보조 → 실패 시 DetectionError. 반환 {x,y,theta}(m·rad)."""
    d = detect_aruco(rgb, dict_name)
    used = "aruco"
    if d is None:
        d = detect_contour(rgb)
        used = "contour"
    if d is None:
        raise DetectionError("ArUco·컨투어 모두 부품 검출 실패")
    u, v = d["center_px"]
    x, y = pixel_to_plane(u, v, calib)
    theta = _wrap(calib["theta_sign"] * (d["theta_img"] - calib["theta_ref"]))
    return {"x": x, "y": y, "theta": theta, "_method": used}


# ---------------------------------------------------------------------------
# self-test (M1 정확도 + 실패 케이스 + M1+M2 닫힌 루프)
# ---------------------------------------------------------------------------
def _import_boot():
    sys.path.insert(0, os.path.join(_HERE, "..", "sim"))
    import boot
    return boot


def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")  # cp949 콘솔 비호환 글리프 크래시 방지
    except Exception:
        pass
    import m2_align
    boot = _import_boot()
    model_path = os.environ.get(
        "PANDA_XML",
        os.path.join("mujoco_menagerie", "franka_emika_panda", "scene.xml"))

    marker_png = scene_parts.generate_aruco_png(os.path.join(_HERE, "aruco_marker.png"))
    model, data, info = scene_parts.build_scene_with_part(model_path, marker_png)
    boot.reset_to_home(model, data)
    camera = boot.resolve_top_camera(model)
    renderer = boot.make_renderer(model, 640, 480)

    try:
        set_pose = lambda p: scene_parts.set_part_pose(model, data, p, info)
        render_fn = lambda: boot.render_rgb(model, data, camera, renderer)

        # 1) 캘리브
        calib = calibrate_scale(set_pose, render_fn)
        print(f"[캘리브] scale={calib['scale']:.6f} m/px, u0={calib['u0']:.1f}, "
              f"v0={calib['v0']:.1f}, theta_sign={calib['theta_sign']:+.0f}")

        # 2) 알려진 pose 검출 정확도 (ground-truth 대비)
        print("[M1 정확도] (known → detected, err)")
        for gt in [{"x": 0.30, "y": 0.45, "theta": 0.0},
                   {"x": 0.45, "y": 0.42, "theta": 0.30},
                   {"x": 0.15, "y": 0.50, "theta": -0.40}]:
            set_pose(gt)
            est = estimate_pose(render_fn(), calib)
            ex, ey, et = est["x"] - gt["x"], est["y"] - gt["y"], _wrap(est["theta"] - gt["theta"])
            print(f"  known=({gt['x']:.2f},{gt['y']:.2f},{gt['theta']:+.2f}) "
                  f"det=({est['x']:.3f},{est['y']:.3f},{est['theta']:+.3f}) [{est['_method']}] "
                  f"err=({ex:+.3f},{ey:+.3f},{et:+.3f})")
            assert abs(ex) < 0.01 and abs(ey) < 0.01, "위치 검출 오차 > 1cm"
            assert abs(et) < 0.06, "각도 검출 오차 > ~3.4deg"

        # 3) 컨투어 보조 검출기 동작 확인(합성: 흰 배경 + 어두운 사각형)
        synth = np.full((480, 640, 3), 255, np.uint8)
        cv2.rectangle(synth, (300, 210), (360, 270), (20, 20, 20), -1)
        cd = detect_contour(synth)
        assert cd is not None and abs(cd["center_px"][0] - 330) < 5 \
            and abs(cd["center_px"][1] - 240) < 5, "컨투어 보조 검출 실패"
        print(f"[컨투어 보조] 합성 사각형 center_px="
              f"{tuple(round(x, 1) for x in cd['center_px'])} (기대 ~330,240)")

        # 4) 검출 실패 케이스(균일 이미지) -> ArUco/컨투어 모두 실패 -> DetectionError
        blank = np.full((480, 640, 3), 127, np.uint8)
        try:
            estimate_pose(blank, calib)
            raise AssertionError("균일 이미지인데 DetectionError 미발생")
        except DetectionError:
            print("[실패 케이스] 균일 이미지 -> ArUco/컨투어 모두 실패 -> DetectionError 정상")

        # 5) M1 + M2 닫힌 루프: correction 적용 시 error 감소(인계 #1 검증)
        target = {"x": 0.30, "y": 0.50, "theta": 0.0}
        set_pose({"x": 0.50, "y": 0.40, "theta": 0.35})  # 초기 오프셋(빈 영역, y>=0.4 유지)
        # tolerance 는 검출 노이즈 바닥(각도 ~0.016rad) 위로 설정.
        conv_tol = 0.03
        print(f"[닫힌 루프] target=(0.30,0.50,0.00), gain=0.6, tol={conv_tol}")
        prev = None
        for cyc in range(8):
            est = estimate_pose(render_fn(), calib)
            align = m2_align.compute_alignment(est, target, gain=0.6, tolerance=conv_tol)
            mag = math.sqrt(sum(v ** 2 for v in align["error"].values()))
            print(f"  cyc={cyc} est=({est['x']:.3f},{est['y']:.3f},{est['theta']:+.3f}) "
                  f"err_norm={mag:.4f} within_tol={align['within_tolerance']}")
            if prev is not None:
                assert mag < prev + 1e-9, "error_norm 이 감소하지 않음(보정 방향 오류)"
            if align["within_tolerance"]:
                break
            prev = mag
            cur = scene_parts.get_part_pose(model, data, info)
            set_pose({"x": cur["x"] + align["correction"]["dx"],
                      "y": cur["y"] + align["correction"]["dy"],
                      "theta": cur["theta"] + align["correction"]["dtheta"]})
        assert mag < 0.05, "닫힌 루프가 충분히 수렴하지 않음"
        print(f"[검증] 닫힌 루프 수렴 OK (err_norm {prev if prev else mag:.4f} → {mag:.4f})")
    finally:
        renderer.close()

    print("S3 VISION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
