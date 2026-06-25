#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (S6 실험·로깅 — 외란하 강건성 통계)
=============================================
외란 종류·강도 × 보정 ON/OFF 조합을 N회씩 반복 실행하고, KPI 4종을 CSV로 누적한다.
  KPI: 잔류 정렬 오차(mm·deg), 정렬 성공률(%), 수렴 사이클 수(보정 반복), 수렴 시간(초).

핵심 배선(S5 인계#1): run_cycle.align_loop 의 ALIGN 측정 렌더를 disturbance.disturbed_render
로 교체(render_fn 주입). 보정 OFF 는 run_cycle(apply_correction=False) 대조군.
D3 매핑(확정): cycle = '정렬 시도 회차(trial idx)', 한 시도 내 iter 에서는 고정 → 회차↑ 드리프트↑.
측정·실제(gt) 오차 둘 다 기록(인계#2), 검출 실패는 ABORT 로 KPI 에 포함(인계#4).

매 시도 레코드(제약): {이미지경로, 정답 pose, 추정 pose, 편차, 외란조건, 사이클, 시간, 성공여부}.

의존: S1 boot, S3 scene_parts/m1_pose, S2 interface, S4 run_cycle, S5 disturbance.
제약: mujoco/numpy/opencv-python + 표준(csv/time/argparse)만, ROS2 미사용, 비ASCII 우회 유지.

실행 방법:
  set PANDA_XML=  (franka_emika_panda/scene.xml 의 경로)
  python experiments/run_experiment.py --disturb d2 --levels off,med --correction both --trials 2
  → results.csv 누적 + summary.csv + 조건별 KPI 표 + "S6 EXPERIMENT: PASS".
  인자: --disturb {none,d1,d2,d3} --levels off,weak,med,strong --correction {on,off,both}
        --trials N --seed S --out DIR --overwrite
"""

import os
import sys
import csv
import math
import time
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "sim"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "external_module"))

import boot          # noqa: E402
import scene_parts   # noqa: E402
import m1_pose       # noqa: E402
import interface     # noqa: E402
import run_cycle     # noqa: E402
import disturbance   # noqa: E402


ALIGN_RES = run_cycle.ALIGN_RES
SOURCE = {"x": 0.20, "y": 0.45, "theta": 0.0}
TARGET = {"x": 0.40, "y": 0.45, "theta": 0.0}

KPI_FIELDS = [
    "trial_idx", "channel", "d1", "d2", "d3", "correction", "image_path",
    "target_x", "target_y", "target_theta",        # 정답 pose
    "est_x", "est_y", "est_theta",                  # 추정 pose(M1)
    "err_dx_mm", "err_dy_mm", "err_dtheta_deg",     # 편차(측정)
    "gt_dx_mm", "gt_dy_mm", "gt_dtheta_deg",        # 편차(실제)
    "cycle", "iters", "time_s",                     # 외란조건/사이클/시간
    "residual_pos_mm", "residual_ang_deg",          # KPI 잔류오차(측정)
    "gt_residual_pos_mm", "gt_residual_ang_deg",
    "success", "success_gt", "state",               # 성공여부
]


def make_render_fn(model, data, camera, renderer, dist_config, cycle):
    """() -> disturbed_render(..., cycle) — D3 는 시도 회차 고정."""
    return lambda: disturbance.disturbed_render(model, data, camera, renderer, dist_config, cycle)


def _err_mm_deg(err):
    return (math.hypot(err["dx"], err["dy"]) * 1000.0, abs(math.degrees(err["dtheta"])))


def run_trial(model, data, info, camera, renderer, calib,
              channel, dist_config, apply_correction, trial_idx, img_dir) -> dict:
    """1회 시도: 외란 render_fn 주입 + 시간측정 + run_cycle + 이미지 저장 → 레코드."""
    render_fn = make_render_fn(model, data, camera, renderer, dist_config, trial_idx)
    t0 = time.perf_counter()
    res = run_cycle.run_cycle(model, data, info, camera, renderer, calib,
                              SOURCE, TARGET, render_fn=render_fn,
                              apply_correction=apply_correction)
    dt = time.perf_counter() - t0

    corr_tag = "on" if apply_correction else "off"
    cond = (f"{channel}_d1{dist_config['d1']}_d2{dist_config['d2']}_d3{dist_config['d3']}"
            f"_corr{corr_tag}_t{trial_idx}")
    img_path = os.path.join(img_dir, cond + ".png")
    try:
        boot.save_rgb(render_fn(), img_path)
    except Exception:
        img_path = ""   # 검출과 무관하게 저장 실패 시 빈 경로

    rec = {k: "" for k in KPI_FIELDS}
    rec.update({
        "trial_idx": trial_idx, "channel": channel,
        "d1": dist_config["d1"], "d2": dist_config["d2"], "d3": dist_config["d3"],
        "correction": corr_tag, "image_path": img_path,
        "target_x": TARGET["x"], "target_y": TARGET["y"], "target_theta": TARGET["theta"],
        "cycle": trial_idx, "iters": res["iters"], "time_s": round(dt, 3),
        "state": res["state"],
    })

    if res["state"] == "DONE" and res["after"] is not None:
        after, gt = res["after"], res["after_gt"]
        rpos, rang = _err_mm_deg(after)
        grpos, grang = _err_mm_deg(gt)
        rec.update({
            "est_x": round(TARGET["x"] - after["dx"], 5),
            "est_y": round(TARGET["y"] - after["dy"], 5),
            "est_theta": round(TARGET["theta"] - after["dtheta"], 5),
            "err_dx_mm": round(after["dx"] * 1000, 3), "err_dy_mm": round(after["dy"] * 1000, 3),
            "err_dtheta_deg": round(math.degrees(after["dtheta"]), 3),
            "gt_dx_mm": round(gt["dx"] * 1000, 3), "gt_dy_mm": round(gt["dy"] * 1000, 3),
            "gt_dtheta_deg": round(math.degrees(gt["dtheta"]), 3),
            "residual_pos_mm": round(rpos, 3), "residual_ang_deg": round(rang, 3),
            "gt_residual_pos_mm": round(grpos, 3), "gt_residual_ang_deg": round(grang, 3),
            "success": int(run_cycle.within_srs(after)),
            "success_gt": int(run_cycle.within_srs(gt)),
        })
    else:  # ABORT(검출 실패) — 실패로 기록(잔류오차는 측정 불가)
        rec.update({"success": 0, "success_gt": 0})
    return rec


def build_matrix(channel: str, levels: list, correction: list, trials: int) -> list:
    """조건 격자: level × correction × trials."""
    matrix = []
    for lv in levels:
        for corr in correction:
            for t in range(trials):
                matrix.append({"channel": channel, "level": lv,
                               "correction": corr, "trial": t})
    return matrix


def append_rows(rows: list, csv_path: str, header: list, overwrite: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if overwrite and os.path.isfile(csv_path):
        os.remove(csv_path)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def summarize(rows: list) -> list:
    """조건(channel,level,correction)별 KPI 집계."""
    groups = {}
    for r in rows:
        key = (r["channel"], r["d1"], r["d2"], r["d3"], r["correction"])
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        n = len(rs)
        done = [r for r in rs if r["state"] == "DONE"]
        aborts = sum(1 for r in rs if r["state"] == "ABORT")
        succ = sum(int(r["success"]) for r in rs)
        succ_gt = sum(int(r["success_gt"]) for r in rs)

        def _mean(field, src):
            vals = [float(r[field]) for r in src if r[field] != ""]
            return round(sum(vals) / len(vals), 3) if vals else ""
        out.append({
            "channel": key[0], "d1": key[1], "d2": key[2], "d3": key[3], "correction": key[4],
            "n": n, "success_rate_%": round(100.0 * succ / n, 1),
            "success_gt_%": round(100.0 * succ_gt / n, 1),
            "abort_rate_%": round(100.0 * aborts / n, 1),
            "mean_residual_mm": _mean("residual_pos_mm", done),
            "mean_residual_deg": _mean("residual_ang_deg", done),
            "mean_iters": _mean("iters", rs),
            "mean_time_s": _mean("time_s", rs),
        })
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="S6 외란하 강건성 실험 러너")
    p.add_argument("--disturb", default="d2", choices=["none", "d1", "d2", "d3"])
    p.add_argument("--levels", default="off,med")
    p.add_argument("--correction", default="both", choices=["on", "off", "both"])
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=_THIS_DIR)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    args = parse_args(argv)
    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    correction = {"on": [True], "off": [False], "both": [True, False]}[args.correction]

    model_path = os.environ.get(
        "PANDA_XML",
        os.path.join("mujoco_menagerie", "franka_emika_panda", "scene.xml"))
    marker_png = scene_parts.generate_aruco_png(
        os.path.join(_THIS_DIR, "..", "external_module", "aruco_marker.png"))
    model, data, info = scene_parts.build_scene_with_part(model_path, marker_png)
    boot.reset_to_home(model, data)
    camera = boot.resolve_top_camera(model)
    renderer = boot.make_renderer(model, ALIGN_RES[0], ALIGN_RES[1])

    img_dir = os.path.join(args.out, "run_images")
    os.makedirs(img_dir, exist_ok=True)
    results_csv = os.path.join(args.out, "results.csv")
    summary_csv = os.path.join(args.out, "summary.csv")

    rows = []
    try:
        # clean calib(외란 없는 기준) — 측정의 기준 좌표계 고정
        set_pose = lambda p: scene_parts.set_part_pose(model, data, p, info)
        calib = m1_pose.calibrate_scale(set_pose, lambda: boot.render_rgb(model, data, camera, renderer))
        interface.configure_perception(calib)
        print(f"[준비] calib scale={calib['scale']*1000:.4f} mm/px, "
              f"매트릭스: disturb={args.disturb} levels={levels} "
              f"correction={args.correction} trials={args.trials}")

        matrix = build_matrix(args.disturb, levels, correction, args.trials)
        for i, cond in enumerate(matrix):
            ch = cond["channel"]
            kw = {ch: cond["level"]} if ch != "none" else {}
            dist_config = disturbance.make_config(seed=args.seed, **kw)
            rec = run_trial(model, data, info, camera, renderer, calib,
                            ch, dist_config, cond["correction"], cond["trial"], img_dir)
            rows.append(rec)
            print(f"  [{i+1}/{len(matrix)}] {ch}={cond['level']} "
                  f"corr={'on' if cond['correction'] else 'off'} t{cond['trial']} "
                  f"→ state={rec['state']} iters={rec['iters']} "
                  f"resid={rec['residual_pos_mm']}mm/{rec['residual_ang_deg']}deg "
                  f"success={rec['success']} time={rec['time_s']}s")
    finally:
        renderer.close()

    append_rows(rows, results_csv, KPI_FIELDS, overwrite=args.overwrite)
    summary = summarize(rows)
    append_rows(summary, summary_csv, list(summary[0].keys()), overwrite=True)

    print("\n[조건별 KPI 요약]")
    for s in summary:
        print(f"  {s['channel']} d2={s['d2']} corr={s['correction']}: "
              f"성공률={s['success_rate_%']}%(gt {s['success_gt_%']}%) "
              f"abort={s['abort_rate_%']}% 잔류={s['mean_residual_mm']}mm/"
              f"{s['mean_residual_deg']}deg iters={s['mean_iters']} time={s['mean_time_s']}s")

    # ---- self-test 단언 ----
    assert len(rows) == len(build_matrix(args.disturb, levels, correction, args.trials)), \
        "행 수가 매트릭스 크기와 불일치"
    # 4종 KPI 결측 없음(DONE 시도): 잔류오차·성공·iters·시간
    for r in rows:
        assert r["iters"] != "" and r["time_s"] != "" and r["success"] != "", "KPI 결측"
        if r["state"] == "DONE":
            assert r["residual_pos_mm"] != "" and r["residual_ang_deg"] != "", "DONE 잔류오차 결측"
    # 보정 ON > OFF 대조(동일 외란 레벨에서): 잔류↓ & 성공률↑
    if True in correction and False in correction:
        on = [r for r in rows if r["correction"] == "on" and r["state"] == "DONE"]
        off = [r for r in rows if r["correction"] == "off" and r["state"] == "DONE"]
        if on and off:
            on_res = np.mean([float(r["residual_pos_mm"]) for r in on])
            off_res = np.mean([float(r["residual_pos_mm"]) for r in off])
            on_succ = np.mean([int(r["success"]) for r in on])
            off_succ = np.mean([int(r["success"]) for r in off])
            print(f"\n[대조] 잔류 ON={on_res:.3f}mm < OFF={off_res:.3f}mm, "
                  f"성공률 ON={on_succ*100:.0f}% >= OFF={off_succ*100:.0f}%")
            assert on_res < off_res, "보정 ON 잔류오차가 OFF보다 작지 않음"
            assert on_succ >= off_succ, "보정 ON 성공률이 OFF보다 낮음"

    print(f"\n[산출물] {results_csv} ({len(rows)}행), {summary_csv}, 이미지 {img_dir}")
    print("S6 EXPERIMENT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
