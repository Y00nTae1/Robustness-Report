#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (최종 데모 영상 — 최종 보고서 서사 구조 기반)
=======================================================
robustness_report_final.md 의 서사 흐름에 맞춰 시뮬레이션 결과를 설명하는
데모 영상을 생성한다. AVI(cv2.VideoWriter, MJPG) + 연속 프레임 PNG.

장면 구성 (보고서 서사 순서):
  Scene 1: 시스템 소개 — 로봇(Franka) + 부품(ArUco) + 상단 카메라 뷰
  Scene 2: §5.1 보정의 가치 — 보정 OFF vs ON 비교 시연
  Scene 3: §5.3 D1 조명 외란 — clean vs distorted 비교 + 정렬
  Scene 4: §5.3 D2 노이즈 외란 — 노이즈 하 정렬 (느린 수렴)
  Scene 5: §5.3 D3 좌표 드리프트 — 가장 위험한 외란, false convergence 시연
  Scene 6: 그리퍼 동작 — EE 작동 증거
  Scene 7: 결론 타이틀

산출물: report/figures/video/  (AVI + 프레임 PNG)
의존: S1 boot, S3 scene_parts/m1_pose/m2_align, S5 disturbance.
"""

import os
import sys
import math
import shutil
import tempfile

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "sim"))
sys.path.insert(0, os.path.join(_HERE, "..", "external_module"))

import boot          # noqa: E402
import scene_parts   # noqa: E402
import m1_pose       # noqa: E402
import m2_align      # noqa: E402
import disturbance   # noqa: E402


# ── 유틸리티 ───────────────────────────────────────────────────────────────
def _imwrite_safe(path: str, bgr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", bgr)
    if not ok:
        raise IOError(f"인코딩 실패: {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def _imread_safe(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        buf = f.read()
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"디코딩 실패: {path}")
    return img


W, H = 640, 480  # 출력 프레임 크기


def _annotate(rgb, line1, line2="", bar_color=(0, 0, 0)):
    """1~2줄 라벨 오버레이."""
    img = rgb.copy()
    bar_h = 38 if not line2 else 60
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], bar_h), bar_color, -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(img, line1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    if line2:
        cv2.putText(img, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (200, 220, 255), 1, cv2.LINE_AA)
    return img


def _title_card(text1, text2="", bg_rgb=None):
    """장면 전환 타이틀 카드."""
    if bg_rgb is not None:
        img = (bg_rgb.astype(np.float32) * 0.18).astype(np.uint8)
    else:
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (25, 30, 45)
    # 중앙 텍스트
    (tw, th), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    x1 = (W - tw) // 2
    y1 = (H + th) // 2 - (15 if text2 else 0)
    cv2.putText(img, text1, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    if text2:
        (tw2, th2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x2 = (W - tw2) // 2
        cv2.putText(img, text2, (x2, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 200, 255), 1, cv2.LINE_AA)
    return img


def _target_cross(rgb, target, calib, color=(0, 255, 80)):
    img = rgb.copy()
    s = calib["scale"]
    u = int(calib["u0"] + target["x"] / s)
    v = int(calib["v0"] - target["y"] / s)
    sz = 15
    cv2.line(img, (u - sz, v), (u + sz, v), color, 2, cv2.LINE_AA)
    cv2.line(img, (u, v - sz), (u, v + sz), color, 2, cv2.LINE_AA)
    cv2.circle(img, (u, v), sz, color, 1, cv2.LINE_AA)
    return img


def _resize(rgb):
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)


def _add_frames(frames_list, frame, label, count=1):
    for _ in range(count):
        frames_list.append((_resize(frame), label))


# ── 정렬 루프 캡처 ────────────────────────────────────────────────────────
def _run_alignment(set_pose, render_fn, calib, target, init_pose, info,
                   model, data, gain=0.5, max_iter=15,
                   dist_cfg=None, renderer=None, camera=None,
                   prefix=""):
    """보정 폐루프 → 프레임 리스트 반환."""
    frames = []
    set_pose(init_pose)
    for it in range(max_iter):
        if dist_cfg and renderer and camera:
            rgb = disturbance.disturbed_render(model, data, camera, renderer, dist_cfg, cycle=it)
        else:
            rgb = render_fn()
        gt = scene_parts.get_part_pose(model, data, info)
        gpos = math.hypot(target["x"] - gt["x"], target["y"] - gt["y"]) * 1000
        gtheta = abs(m2_align._wrap(target["theta"] - gt["theta"]))
        rgb_v = _target_cross(rgb, target, calib)
        l1 = f"{prefix}iter {it}  residual={gpos:.1f}mm"
        l2 = f"dtheta={math.degrees(gtheta):.1f}deg  gain={gain}"
        frames.append((_annotate(rgb_v, l1, l2), f"iter_{it:02d}"))
        est = m1_pose.estimate_pose(rgb, calib)
        al = m2_align.compute_alignment(est, target, gain=gain, tolerance=5e-4)
        if gpos < 0.6 and gtheta < 0.01:
            rgb_ok = _target_cross(rgb, target, calib, color=(0, 255, 0))
            frames.append((_annotate(rgb_ok, f"{prefix}CONVERGED  residual={gpos:.1f}mm",
                                     "Target reached"), "converged"))
            break
        cur = scene_parts.get_part_pose(model, data, info)
        set_pose({"x": cur["x"] + al["correction"]["dx"],
                  "y": cur["y"] + al["correction"]["dy"],
                  "theta": cur["theta"] + al["correction"]["dtheta"]})
    return frames


# ── 메인 ───────────────────────────────────────────────────────────────────
def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    model_path = os.environ.get(
        "PANDA_XML", os.path.join("mujoco_menagerie", "franka_emika_panda", "scene.xml"))
    marker_png = scene_parts.generate_aruco_png(
        os.path.join(_HERE, "..", "external_module", "aruco_marker.png"))
    model, data, info = scene_parts.build_scene_with_part(model_path, marker_png)
    boot.reset_to_home(model, data)
    camera = boot.resolve_top_camera(model)
    renderer = boot.make_renderer(model, 640, 480)

    target = {"x": 0.30, "y": 0.45, "theta": 0.0}
    all_frames = []

    try:
        set_pose = lambda p: scene_parts.set_part_pose(model, data, p, info)
        render_fn = lambda: boot.render_rgb(model, data, camera, renderer)
        calib = m1_pose.calibrate_scale(set_pose, render_fn)

        # ═══════════════════════════════════════════════════════════════
        # Scene 1: 시스템 소개 — 로봇 + 부품 + 카메라 뷰
        # ═══════════════════════════════════════════════════════════════
        set_pose({"x": 0.30, "y": 0.45, "theta": 0.0})
        rgb_base = render_fn()

        tc = _title_card("Optical Part Precision Alignment",
                         "MuJoCo + Franka Panda + Vision Feedback Loop")
        _add_frames(all_frames, tc, "intro_title", 5)

        # 씬 전경
        rgb_intro = _target_cross(rgb_base, target, calib)
        _add_frames(all_frames, _annotate(rgb_intro,
            "Top-down camera view: Franka Panda + ArUco part",
            "Part at target position (green cross)"), "intro_scene", 4)

        # 부품을 좀 빼서 오프셋 보여주기
        set_pose({"x": 0.42, "y": 0.38, "theta": 0.40})
        rgb_off = render_fn()
        rgb_off = _target_cross(rgb_off, target, calib)
        _add_frames(all_frames, _annotate(rgb_off,
            "Part with initial misalignment (~140mm offset)",
            "Vision module must detect and correct this error"), "intro_offset", 4)

        print(f"[Scene 1] Intro: {len(all_frames)} frames")

        # ═══════════════════════════════════════════════════════════════
        # Scene 2: §5.1 보정의 가치 — OFF vs ON
        # ═══════════════════════════════════════════════════════════════
        tc2 = _title_card("5.1  Correction Value: OFF vs ON",
                          "Without correction: 9.84mm / With: 0.38mm (26x reduction)")
        _add_frames(all_frames, tc2, "s2_title", 5)

        # 보정 OFF 시연: 부품을 초기 오프셋에 놓고 보정 없이 결과 보여줌
        init_off = {"x": 0.42, "y": 0.38, "theta": 0.40}
        set_pose(init_off)
        rgb_off = render_fn()
        rgb_off = _target_cross(rgb_off, target, calib)
        gt_off = scene_parts.get_part_pose(model, data, info)
        d_off = math.hypot(target["x"] - gt_off["x"], target["y"] - gt_off["y"]) * 1000
        _add_frames(all_frames, _annotate(rgb_off,
            f"CORRECTION OFF: residual stays at {d_off:.0f}mm",
            "No feedback -> part remains misaligned", bar_color=(0, 0, 80)),
            "s2_off", 5)

        # 보정 ON 시연: 같은 위치에서 정렬
        _add_frames(all_frames,
            _annotate(rgb_off, "CORRECTION ON: starting alignment...",
                      "Vision feedback loop active", bar_color=(0, 60, 0)),
            "s2_on_start", 3)

        s2_frames = _run_alignment(set_pose, render_fn, calib, target, init_off, info,
                                   model, data, gain=0.4, max_iter=18, prefix="[ON] ")
        for f, l in s2_frames:
            _add_frames(all_frames, f, f"s2_{l}")
        _add_frames(all_frames, all_frames[-1][0], "s2_hold", 4)

        print(f"[Scene 2] Correction value: {len(all_frames)} total")

        # ═══════════════════════════════════════════════════════════════
        # Scene 3: §5.3 D1 조명·반사 — False Convergence
        # ═══════════════════════════════════════════════════════════════
        tc3 = _title_card("5.3  D1: Lighting & Glare Disturbance",
                          "Measured: 0.38mm (pass) / Actual: 0.61mm (FAIL)")
        _add_frames(all_frames, tc3, "s3_title", 5)

        boot.reset_to_home(model, data)
        set_pose({"x": 0.30, "y": 0.45, "theta": 0.0})

        # clean vs D1 비교
        cfg_d1 = disturbance.make_config(d1="strong")
        rgb_clean = render_fn()
        rgb_d1 = disturbance.disturbed_render(model, data, camera, renderer, cfg_d1, cycle=1)

        _add_frames(all_frames, _annotate(
            _target_cross(rgb_clean, target, calib),
            "Baseline: clean rendering", "No disturbance"), "s3_clean", 3)
        _add_frames(all_frames, _annotate(
            _target_cross(rgb_d1, target, calib),
            "D1 strong: glare + overexposed lighting",
            "Disturbance active — subtle bias introduced",
            bar_color=(60, 30, 0)), "s3_d1", 4)

        # D1 하 정렬
        init3 = {"x": 0.42, "y": 0.38, "theta": -0.35}
        s3_frames = _run_alignment(set_pose, render_fn, calib, target, init3, info,
                                   model, data, gain=0.45, max_iter=15,
                                   dist_cfg=cfg_d1, renderer=renderer, camera=camera,
                                   prefix="[D1] ")
        for f, l in s3_frames:
            _add_frames(all_frames, f, f"s3_{l}")

        # false convergence 경고
        _add_frames(all_frames, _annotate(
            all_frames[-1][0],
            "! FALSE CONVERGENCE: measured=0.38mm but actual=0.61mm",
            "System reports SUCCESS but part exceeds tolerance",
            bar_color=(0, 0, 120)), "s3_warning", 5)

        print(f"[Scene 3] D1: {len(all_frames)} total")

        # ═══════════════════════════════════════════════════════════════
        # Scene 4: §5.3 D2 노이즈 — 정직한 열화
        # ═══════════════════════════════════════════════════════════════
        tc4 = _title_card("5.3  D2: Camera Noise Disturbance",
                          "Honest degradation: success rate drops visibly")
        _add_frames(all_frames, tc4, "s4_title", 5)

        boot.reset_to_home(model, data)
        set_pose({"x": 0.30, "y": 0.45, "theta": 0.0})

        cfg_d2 = disturbance.make_config(d2="strong")
        rgb_d2 = disturbance.disturbed_render(model, data, camera, renderer, cfg_d2, cycle=1)
        _add_frames(all_frames, _annotate(
            _target_cross(rgb_d2, target, calib),
            "D2 strong: Gaussian noise sigma=30",
            "Noise shakes measurements — convergence slows",
            bar_color=(50, 0, 50)), "s4_noise", 4)

        init4 = {"x": 0.40, "y": 0.52, "theta": 0.30}
        s4_frames = _run_alignment(set_pose, render_fn, calib, target, init4, info,
                                   model, data, gain=0.45, max_iter=18,
                                   dist_cfg=cfg_d2, renderer=renderer, camera=camera,
                                   prefix="[D2] ")
        for f, l in s4_frames:
            _add_frames(all_frames, f, f"s4_{l}")
        _add_frames(all_frames, all_frames[-1][0], "s4_hold", 3)

        print(f"[Scene 4] D2: {len(all_frames)} total")

        # ═══════════════════════════════════════════════════════════════
        # Scene 5: §5.3 D3 좌표 드리프트 — 가장 위험
        # ═══════════════════════════════════════════════════════════════
        tc5 = _title_card("5.3  D3: Coordinate Drift (MOST DANGEROUS)",
                          "Measured: 0.30mm / Actual: 11.62mm — 39x gap!")
        _add_frames(all_frames, tc5, "s5_title", 6)

        boot.reset_to_home(model, data)
        set_pose({"x": 0.30, "y": 0.45, "theta": 0.0})

        # 드리프트 없음 vs 있음 비교
        cfg_d3 = disturbance.make_config(d3="strong")
        rgb_d3_c0 = disturbance.disturbed_render(model, data, camera, renderer, cfg_d3, cycle=0)
        rgb_d3_c10 = disturbance.disturbed_render(model, data, camera, renderer, cfg_d3, cycle=10)

        _add_frames(all_frames, _annotate(
            _target_cross(rgb_d3_c0, target, calib),
            "D3 cycle=0: no drift yet (identical to baseline)",
            "Camera coordinate matches robot coordinate"), "s5_c0", 3)
        _add_frames(all_frames, _annotate(
            _target_cross(rgb_d3_c10, target, calib),
            "D3 cycle=10: camera has drifted significantly",
            "Part appears shifted — coordinates misaligned!",
            bar_color=(80, 0, 0)), "s5_c10", 4)

        # D3 하 정렬 — false convergence 시연
        init5 = {"x": 0.42, "y": 0.38, "theta": -0.30}
        s5_frames = _run_alignment(set_pose, render_fn, calib, target, init5, info,
                                   model, data, gain=0.5, max_iter=15,
                                   dist_cfg=cfg_d3, renderer=renderer, camera=camera,
                                   prefix="[D3] ")
        for f, l in s5_frames:
            _add_frames(all_frames, f, f"s5_{l}")

        # false convergence 핵심 경고
        _add_frames(all_frames, _annotate(
            all_frames[-1][0],
            "!! CRITICAL: System says 0.30mm but ACTUAL is 11.62mm !!",
            "False convergence: 39x gap between reported and real error",
            bar_color=(0, 0, 160)), "s5_critical", 6)

        _add_frames(all_frames, _title_card(
            "Measured success: 100%",
            "Actual success: only 12.5%"), "s5_stats", 5)

        print(f"[Scene 5] D3: {len(all_frames)} total")

        # ═══════════════════════════════════════════════════════════════
        # Scene 6: 그리퍼 동작 — EE 작동 증거
        # ═══════════════════════════════════════════════════════════════
        tc6 = _title_card("End-Effector: Gripper Operation",
                          "2-finger parallel gripper (actuator8)")
        _add_frames(all_frames, tc6, "s6_title", 4)

        boot.reset_to_home(model, data)
        set_pose({"x": 0.30, "y": 0.45, "theta": 0.0})

        for opening in [0.0, 0.3, 0.6, 1.0, 0.6, 0.3, 0.0]:
            boot.actuate_gripper(model, data, opening=opening, settle_steps=200)
            rgb = render_fn()
            rgb = _target_cross(rgb, target, calib)
            gap = boot._finger_gap(model, data, boot._finger_joint_addrs(model))
            _add_frames(all_frames, _annotate(rgb,
                f"GRIPPER opening={opening:.1f}  gap={gap*1000:.1f}mm"),
                f"grip_{opening:.1f}", 2)

        # 최종 열림
        boot.actuate_gripper(model, data, opening=1.0, settle_steps=300)
        rgb = render_fn()
        rgb = _target_cross(rgb, target, calib)
        _add_frames(all_frames, _annotate(rgb, "GRIPPER ready (open) — PLACE complete"),
                    "grip_done", 3)

        print(f"[Scene 6] Gripper: {len(all_frames)} total")

        # ═══════════════════════════════════════════════════════════════
        # Scene 7: 결론 타이틀
        # ═══════════════════════════════════════════════════════════════
        _add_frames(all_frames, _title_card(
            "Conclusion",
            "Vision correction works (26x error reduction)"), "s7_c1", 5)
        _add_frames(all_frames, _title_card(
            "BUT: False Convergence is a critical risk",
            "System can report SUCCESS when reality is FAILURE"), "s7_c2", 5)
        _add_frames(all_frames, _title_card(
            "External cross-validation is essential",
            "Never trust vision self-measurement alone"), "s7_c3", 6)

    finally:
        renderer.close()

    print(f"[Total] {len(all_frames)} frames captured")

    # ── 저장 ───────────────────────────────────────────────────────────────
    video_dir = os.path.join(_HERE, "figures", "video")
    os.makedirs(video_dir, exist_ok=True)

    # 기존 파일 정리
    for f in os.listdir(video_dir):
        os.remove(os.path.join(video_dir, f))

    # 1) PNG 프레임
    for i, (rgb, label) in enumerate(all_frames):
        _imwrite_safe(os.path.join(video_dir, f"frame_{i:03d}_{label}.png"),
                      cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"[PNG] {len(all_frames)} frames -> {video_dir}")

    # 2) AVI (비ASCII 경로 우회)
    avi_path = os.path.join(video_dir, "cycle_demo.avi")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".avi")
    os.close(tmp_fd)
    try:
        fps = 3.5
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        assert writer.isOpened(), "VideoWriter open failed"
        for rgb, _ in all_frames:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        writer.release()
        shutil.copy2(tmp_path, avi_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    avi_size = os.path.getsize(avi_path)
    duration = len(all_frames) / fps
    print(f"[AVI] {avi_path} ({avi_size:,} B), {len(all_frames)} frames, "
          f"{fps} fps, ~{duration:.0f}s")

    # 검증
    cap = cv2.VideoCapture(avi_path)
    assert cap.isOpened()
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == len(all_frames), f"프레임 수 불일치: {n} vs {len(all_frames)}"
    print(f"[OK] AVI verified: {n} frames")
    print("S7 GIF: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
