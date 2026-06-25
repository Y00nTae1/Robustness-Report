#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (S5 외란 주입 — 비전 측정 계층)
=========================================
세 가지 현실 외란을 코드로 주입하는 모듈. 강도(off/weak/med/strong)·ON/OFF 토글.
  D1 조명·반사 : 렌더 단계 headlight(diffuse/ambient) 스케일 + 이미지 반사 글레어.
  D2 카메라 노이즈: 렌더 RGB 에 가우시안 가산(σ 레벨).
  D3 좌표 드리프트: cycle 비례로 카메라 외부 파라미터(lookat) 오프셋 → 측정 bias 유발.

범위(S4 인계#1 확정): 외란은 '비전 측정 계층'에만 주입한다(부품 mocap 추상화 유지,
물리 파지로 승격하지 않음). off 조합은 baseline 과 비트 동일(항등성).

의존: S1 boot(make_renderer/render_rgb/resolve_top_camera), S3 scene_parts(부품 씬),
      m1_pose(외란 영향 관찰용). 외부 lib: mujoco/numpy(+표준), 저장은 boot.save_rgb(imencode).
제약: ROS2 미사용, 비ASCII 경로 우회 유지, 공식 모델 무수정.

실행 방법(단독 self-test):
  set PANDA_XML=  (franka_emika_panda/scene.xml 의 경로)
  python sim/disturbance.py
  -> OFF=baseline 항등성 + D1/D2/D3 레벨별 단조 효과 + 복원 확인 + 샘플 저장 후
     "S5 DISTURB: PASS". 산출물: experiments/disturbance_samples/*.png
"""

import os
import sys

import numpy as np
import mujoco

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "external_module"))

import boot          # noqa: E402  (S1)
import scene_parts   # noqa: E402  (S3)
import m1_pose       # noqa: E402  (S3, 영향 관찰용)


# 레벨 → 수치 (한 곳에 고정: 재현성·문서화)
LEVELS = {
    "d1":       {"off": 1.0, "weak": 1.3,  "med": 1.7,  "strong": 2.2},   # headlight 스케일
    "d1_glare": {"off": 0.0, "weak": 0.15, "med": 0.30, "strong": 0.5},   # 반사 글레어 강도
    "d2":       {"off": 0.0, "weak": 5.0,  "med": 15.0, "strong": 30.0},  # 노이즈 σ(uint8)
    "d3":       {"off": 0.0, "weak": 5e-4, "med": 1.5e-3, "strong": 3e-3},# m/cycle lookat 드리프트
}
_VALID = ("off", "weak", "med", "strong")


# ---------------------------------------------------------------------------
# 설정/토글
# ---------------------------------------------------------------------------
def make_config(d1: str = "off", d2: str = "off", d3: str = "off", seed: int = 0) -> dict:
    """외란 토글+강도 설정. 각 레벨은 off/weak/med/strong."""
    for name, lv in (("d1", d1), ("d2", d2), ("d3", d3)):
        if lv not in _VALID:
            raise ValueError(f"{name} 레벨이 잘못됨: {lv!r} (가능: {_VALID})")
    return {"d1": d1, "d2": d2, "d3": d3, "seed": int(seed)}


# ---------------------------------------------------------------------------
# D1 조명·반사
# ---------------------------------------------------------------------------
def apply_lighting(model, level: str) -> dict:
    """headlight diffuse/ambient 를 레벨 배율로 스케일(렌더 조명 변화). 변경전 상태 반환."""
    scale = LEVELS["d1"][level]
    hl = model.vis.headlight
    saved = {"diffuse": np.array(hl.diffuse), "ambient": np.array(hl.ambient)}
    hl.diffuse[:] = np.clip(saved["diffuse"] * scale, 0.0, 1.0)
    hl.ambient[:] = np.clip(saved["ambient"] * scale, 0.0, 1.0)
    return saved


def restore_lighting(model, saved: dict) -> None:
    """apply_lighting 변경을 원복."""
    hl = model.vis.headlight
    hl.diffuse[:] = saved["diffuse"]
    hl.ambient[:] = saved["ambient"]


def add_glare(rgb: np.ndarray, strength: float, rng) -> np.ndarray:
    """반사 모사: 밝은 원형 spot(가우시안 falloff)을 합성. strength<=0 이면 무변화."""
    if strength <= 0:
        return rgb
    h, w = rgb.shape[:2]
    cx = int(rng.integers(w // 4, 3 * w // 4))
    cy = int(rng.integers(h // 4, 3 * h // 4))
    Y, X = np.ogrid[:h, :w]
    r = np.hypot(X - cx, Y - cy)
    rad = 0.22 * min(h, w)
    falloff = np.clip(1.0 - r / rad, 0.0, 1.0) ** 2
    add = (falloff * strength * 255.0)[..., None]
    return np.clip(rgb.astype(np.float32) + add, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# D2 카메라 가우시안 노이즈
# ---------------------------------------------------------------------------
def add_gaussian_noise(rgb: np.ndarray, sigma: float, rng) -> np.ndarray:
    """RGB 에 가우시안 노이즈(σ) 가산, 0~255 클립. σ<=0 이면 무변화."""
    if sigma <= 0:
        return rgb
    noise = rng.normal(0.0, sigma, size=rgb.shape)
    return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# D3 좌표 드리프트
# ---------------------------------------------------------------------------
def drift_camera(base_camera, level: str, cycle: int):
    """카메라 외부(lookat)에 cycle 비례 오프셋을 준 복사본 반환(base 불변)."""
    d = LEVELS["d3"][level] * cycle
    c = mujoco.MjvCamera()
    c.type = base_camera.type
    c.fixedcamid = base_camera.fixedcamid
    c.trackbodyid = base_camera.trackbodyid
    c.distance = base_camera.distance
    c.azimuth = base_camera.azimuth
    c.elevation = base_camera.elevation
    c.orthographic = base_camera.orthographic
    c.lookat[:] = np.array(base_camera.lookat)
    c.lookat[0] += d            # x 방향 드리프트
    c.lookat[1] += d * 0.5      # y 방향 동반 드리프트(2D bias)
    return c


# ---------------------------------------------------------------------------
# 통합 render 래퍼
# ---------------------------------------------------------------------------
def disturbed_render(model, data, camera, renderer, config: dict, cycle: int = 0) -> np.ndarray:
    """D3(카메라)→렌더(D1 조명)→D1 글레어→D2 노이즈. off 조합은 baseline 과 동일."""
    rng = np.random.default_rng(config["seed"] * 100003 + cycle)

    cam = drift_camera(camera, config["d3"], cycle) if config["d3"] != "off" else camera

    if config["d1"] != "off":
        saved = apply_lighting(model, config["d1"])
        try:
            rgb = boot.render_rgb(model, data, cam, renderer)
        finally:
            restore_lighting(model, saved)
        rgb = add_glare(rgb, LEVELS["d1_glare"][config["d1"]], rng)
    else:
        rgb = boot.render_rgb(model, data, cam, renderer)

    if config["d2"] != "off":
        rgb = add_gaussian_noise(rgb, LEVELS["d2"][config["d2"]], rng)
    return rgb


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _center(rgb):
    d = m1_pose.detect_aruco(rgb)
    return None if d is None else np.array(d["center_px"])


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
    scene_parts.set_part_pose(model, data, {"x": 0.30, "y": 0.45, "theta": 0.0}, info)
    camera = boot.resolve_top_camera(model)
    renderer = boot.make_renderer(model, 640, 480)

    outdir = os.path.join(_THIS_DIR, "..", "experiments", "disturbance_samples")
    os.makedirs(outdir, exist_ok=True)

    try:
        # 0) 항등성: OFF == baseline (비트 동일)
        base = disturbed_render(model, data, camera, renderer, make_config())
        base_ref = boot.render_rgb(model, data, camera, renderer)
        assert np.array_equal(base, base_ref), "OFF 가 baseline 과 다름(항등성 위배)"
        boot.save_rgb(base, os.path.join(outdir, "baseline.png"))
        bmean = float(base.mean())
        bc = _center(base)
        print(f"[항등성] OFF == baseline OK (mean={bmean:.1f})")

        # 1) D1 조명·반사: 밝기 단조 증가
        print("[D1 조명] level: mean brightness / 포화픽셀%")
        prev = bmean
        for lv in ("weak", "med", "strong"):
            rgb = disturbed_render(model, data, camera, renderer, make_config(d1=lv))
            mb = float(rgb.mean()); sat = float((rgb >= 250).mean() * 100)
            print(f"  {lv}: mean={mb:.1f} (base {bmean:.1f}), 포화={sat:.2f}%")
            assert mb > bmean, "D1 이 밝기를 올리지 않음"
            assert mb >= prev - 0.5, "D1 밝기 단조성 위배"
            prev = mb
            boot.save_rgb(rgb, os.path.join(outdir, f"d1_{lv}.png"))
        # 복원: D1 이후 일반 렌더가 baseline 과 동일해야
        assert np.array_equal(boot.render_rgb(model, data, camera, renderer), base_ref), \
            "D1 후 조명 복원 실패"
        print("[D1 복원] headlight 원복 OK")

        # 2) D2 노이즈: baseline 대비 표준편차 단조 증가
        print("[D2 노이즈] level: 측정 std (목표 σ)")
        prev = 0.0
        for lv in ("weak", "med", "strong"):
            rgb = disturbed_render(model, data, camera, renderer, make_config(d2=lv))
            std = float((rgb.astype(np.float32) - base.astype(np.float32)).std())
            print(f"  {lv}: std={std:.2f} (σ={LEVELS['d2'][lv]})")
            assert std > prev, "D2 노이즈 단조성 위배"
            prev = std
            boot.save_rgb(rgb, os.path.join(outdir, f"d2_{lv}.png"))

        # 3) D3 드리프트: cycle=0 무변화, cycle=10 중심 이동 단조 증가
        d3c0 = disturbed_render(model, data, camera, renderer, make_config(d3="strong"), cycle=0)
        assert np.allclose(_center(d3c0), bc, atol=0.5), "D3 cycle=0 인데 변화 발생"
        print("[D3 드리프트] cycle=0 무변화 OK / cycle=10 중심 이동(px):")
        prev = 0.0
        for lv in ("weak", "med", "strong"):
            rgb = disturbed_render(model, data, camera, renderer, make_config(d3=lv), cycle=10)
            shift = float(np.linalg.norm(_center(rgb) - bc))
            print(f"  {lv}: 중심 이동={shift:.2f}px")
            assert shift > prev, "D3 드리프트 단조성 위배"
            prev = shift
            boot.save_rgb(rgb, os.path.join(outdir, f"d3_{lv}_cyc10.png"))

        # 4) 외란하 M1 측정 영향 1샷(clean calib 기준)
        set_pose = lambda p: scene_parts.set_part_pose(model, data, p, info)
        calib = m1_pose.calibrate_scale(set_pose, lambda: boot.render_rgb(model, data, camera, renderer))
        set_pose({"x": 0.30, "y": 0.45, "theta": 0.0})
        est_off = m1_pose.estimate_pose(
            disturbed_render(model, data, camera, renderer, make_config()), calib)
        est_on = m1_pose.estimate_pose(
            disturbed_render(model, data, camera, renderer, make_config(d1="med", d2="med")), calib)
        dpos = np.hypot(est_on["x"] - est_off["x"], est_on["y"] - est_off["y"]) * 1000
        print(f"[M1 영향] off=({est_off['x']:.4f},{est_off['y']:.4f})  "
              f"D1+D2(med)=({est_on['x']:.4f},{est_on['y']:.4f})  측정변화={dpos:.3f}mm")
    finally:
        renderer.close()

    print(f"[산출물] 샘플 저장: {outdir}")
    print("S5 DISTURB: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
