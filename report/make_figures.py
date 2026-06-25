#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (S7 리포트 채움 — CSV 집계 기반 KPI 그래프 + 완성 리포트)
================================================================
experiments/results.csv(per-trial)를 **실제로 읽어** 조건별(channel, 강도, correction)로
집계하고, 그 집계값으로만 KPI 그래프 3종(opencv PNG)과 과제 필수 5항목 리포트를 채운다.
하드코딩 통계 없음 — 모든 숫자는 CSV 집계에서 나온다(추측·생성 금지).

집계 키: (channel, level, correction). level = 해당 채널 컬럼값(d1/d2/d3), none 채널은 'off'.
집계 지표: n, 측정성공%, 실제성공%(success_gt), 측정잔류mm, 실제잔류mm, iters, time_s, ABORT수.
  (DONE 행만 잔류오차 평균, 빈 셀 제외.)

그래프(요청 매핑):
  G1 외란-성능 곡선   : x=강도, y=measured% & ground-truth%(오버레이, correction=on) — false-convergence.
  G2 보정 ON/OFF 비교 : 막대 — success% / mean residual(mm) (baseline none).
  G3 정확도-속도       : 산점 — per-trial time_s vs residual_pos_mm, 색=correction.

matplotlib 미사용: cv2 프리미티브로 렌더, 영문 라벨. 저장은 imencode+파일IO(비ASCII 안전).
라이브러리: numpy, opencv-python + 표준(os/re/csv). ROS2 미사용.
실행: python report/make_figures.py
"""

import os
import re
import csv
import math

import numpy as np
import cv2


_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(_HERE, "figures")
RESULTS_CSV = os.path.join(_HERE, "..", "experiments", "results.csv")

SRS = {"clearance_mm": 0.5, "clearance_deg": 2.0, "gsd_mm_px": 1.4475, "align_res": "1280x960"}
LEVELS = ["weak", "med", "strong"]
BLUE, RED, GREEN, GRAY, BLACK = (200, 80, 0), (40, 40, 220), (40, 160, 40), (160, 160, 160), (0, 0, 0)


# ── CSV 로드 & 집계 ───────────────────────────────────────────────────────────
def load_csv(path: str) -> list:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"results.csv 없음: {path} — S6 캠페인을 먼저 실행하세요.")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _row_level(r: dict) -> str:
    ch = r["channel"]
    return r[ch] if ch in ("d1", "d2", "d3") else "off"


def _fmean(field, src):
    vals = [float(r[field]) for r in src if str(r.get(field, "")).strip() not in ("", "nan")]
    return (sum(vals) / len(vals)) if vals else float("nan")


def aggregate(rows: list) -> dict:
    """(channel, level, correction) -> 집계 dict."""
    groups = {}
    for r in rows:
        groups.setdefault((r["channel"], _row_level(r), r["correction"]), []).append(r)
    agg = {}
    for key, rs in groups.items():
        n = len(rs)
        done = [r for r in rs if r["state"] == "DONE"]
        agg[key] = {
            "n": n,
            "meas_succ": 100.0 * sum(int(r["success"]) for r in rs) / n,
            "gt_succ": 100.0 * sum(int(r["success_gt"]) for r in rs) / n,
            "meas_res": _fmean("residual_pos_mm", done),
            "gt_res": _fmean("gt_residual_pos_mm", done),
            "iters": _fmean("iters", rs),
            "time": _fmean("time_s", rs),
            "abort": sum(1 for r in rs if r["state"] == "ABORT"),
        }
    return agg


def print_table(agg: dict) -> None:
    print("\n[CSV 집계 통계표]  (출처: experiments/results.csv)")
    print(f"  {'channel':7} {'level':7} {'corr':4} {'n':>3} "
          f"{'meas%':>6} {'gt%':>6} {'mres_mm':>8} {'gtres_mm':>9} {'iters':>6} {'time_s':>7} {'abort':>5}")
    order = {"off": 0, "weak": 1, "med": 2, "strong": 3}
    for key in sorted(agg, key=lambda k: (k[0], order.get(k[1], 9), k[2])):
        ch, lv, co = key; a = agg[key]
        print(f"  {ch:7} {lv:7} {co:4} {a['n']:>3} "
              f"{a['meas_succ']:>6.1f} {a['gt_succ']:>6.1f} "
              f"{a['meas_res']:>8.2f} {a['gt_res']:>9.2f} "
              f"{a['iters']:>6.1f} {a['time']:>7.3f} {a['abort']:>5}")


# ── 그래프 공통 ───────────────────────────────────────────────────────────────
def save_png(canvas, out_path) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise IOError(f"PNG 인코딩 실패: {out_path}")
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    return out_path


def _text(img, s, org, scale=0.45, color=BLACK, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _new_canvas(w, h, title):
    img = np.full((h, w, 3), 255, np.uint8)
    _text(img, title, (12, 24), 0.6, BLACK, 2)
    return img


def _ymap(v, ymin, ymax, y0, h):
    v = max(min(v, ymax), ymin)
    return int(y0 + h - (v - ymin) / (ymax - ymin) * h)


def _axes(img, x0, y0, w, h, ymin, ymax, x_labels, y_label, ticks):
    cv2.line(img, (x0, y0), (x0, y0 + h), BLACK, 1)
    cv2.line(img, (x0, y0 + h), (x0 + w, y0 + h), BLACK, 1)
    for fr in ticks:
        v = ymin + fr * (ymax - ymin)
        py = _ymap(v, ymin, ymax, y0, h)
        cv2.line(img, (x0 - 3, py), (x0, py), BLACK, 1)
        _text(img, f"{v:.0f}" if ymax > 5 else f"{v:.1f}", (x0 - 36, py + 4), 0.4)
    n = len(x_labels)
    xs = [x0 + int((i + 0.5) / n * w) for i in range(n)]
    for i, lab in enumerate(x_labels):
        _text(img, lab, (xs[i] - 16, y0 + h + 16), 0.4)
    _text(img, y_label, (x0 - 36, y0 - 8), 0.4)
    return xs


# ── G1/G2/G3 (집계·per-trial 기반) ───────────────────────────────────────────
def fig_disturbance_curve(agg, out_path) -> str:
    img = _new_canvas(1200, 470, "G1  Disturbance vs Performance (measured vs ground-truth success %, corr=ON)")
    pw, ph, top = 330, 320, 70
    for ci, ch in enumerate(["d1", "d2", "d3"]):
        x0 = 80 + ci * 380
        xs = _axes(img, x0, top, pw, ph, 0, 100, LEVELS, "succ%", (0, 0.25, 0.5, 0.75, 1.0))
        meas = [agg.get((ch, lv, "on"), {}).get("meas_succ", float("nan")) for lv in LEVELS]
        gt = [agg.get((ch, lv, "on"), {}).get("gt_succ", float("nan")) for lv in LEVELS]
        _text(img, ch.upper(), (x0 + pw // 2 - 10, top - 12), 0.55, BLACK, 2)
        for vals, color in ((meas, BLUE), (gt, RED)):
            pts = [(xs[i], _ymap(vals[i], 0, 100, top, ph)) for i in range(len(LEVELS))]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(img, a, b, color, 2)
            for p, v in zip(pts, vals):
                cv2.circle(img, p, 4, color, -1)
                _text(img, f"{v:.0f}", (p[0] + 4, p[1] - 6), 0.4, color)
        gap = meas[-1] - gt[-1]
        if gap > 5:
            _text(img, f"gap={gap:.0f}%", (x0 + pw - 95, top + 18), 0.45, RED)
    cv2.line(img, (80, 440), (110, 440), BLUE, 2); _text(img, "measured success%", (115, 444), 0.45, BLUE)
    cv2.line(img, (350, 440), (380, 440), RED, 2); _text(img, "ground-truth success% (real)", (385, 444), 0.45, RED)
    return save_png(img, out_path)


def fig_correction_compare(agg, out_path) -> str:
    img = _new_canvas(760, 440, "G2  Correction ON vs OFF (baseline, no disturbance)")
    on = agg[("none", "off", "on")]; off = agg[("none", "off", "off")]
    _axes(img, 80, 70, 250, 300, 0, 100, ["ON", "OFF"], "succ%", (0, 0.5, 1.0))
    for i, (val, col) in enumerate(((on["meas_succ"], GREEN), (off["meas_succ"], RED))):
        bx = 80 + 40 + i * 110; by = _ymap(val, 0, 100, 70, 300)
        cv2.rectangle(img, (bx, by), (bx + 60, 370), col, -1); _text(img, f"{val:.0f}%", (bx + 8, by - 6), 0.5)
    _text(img, "Success rate", (140, 60), 0.5)
    rmax = max(off["meas_res"], 1.0) * 1.1
    _axes(img, 470, 70, 250, 300, 0, rmax, ["ON", "OFF"], "mm", (0, 0.5, 1.0))
    for i, (val, col) in enumerate(((on["meas_res"], GREEN), (off["meas_res"], RED))):
        bx = 470 + 40 + i * 110; by = _ymap(val, 0, rmax, 70, 300)
        cv2.rectangle(img, (bx, by), (bx + 60, 370), col, -1); _text(img, f"{val:.2f}", (bx + 4, by - 6), 0.5)
    _text(img, "Mean residual (mm)", (510, 60), 0.5)
    _text(img, f"ON: {off['meas_res']:.2f}mm -> {on['meas_res']:.2f}mm,  "
               f"{off['meas_succ']:.0f}% -> {on['meas_succ']:.0f}%", (80, 425), 0.5)
    return save_png(img, out_path)


def fig_tradeoff(rows, out_path) -> str:
    img = _new_canvas(760, 470, "G3  Accuracy-Speed Tradeoff (per-trial time vs residual, color=correction)")
    x0, y0, w, h = 80, 70, 600, 330
    done = [r for r in rows if r["state"] == "DONE" and str(r.get("residual_pos_mm", "")).strip() != ""]
    tmax = max(float(r["time_s"]) for r in done) * 1.1
    rmax = max(float(r["residual_pos_mm"]) for r in done) * 1.1
    cv2.line(img, (x0, y0), (x0, y0 + h), BLACK, 1); cv2.line(img, (x0, y0 + h), (x0 + w, y0 + h), BLACK, 1)
    for fr in (0, 0.5, 1.0):
        px = x0 + int(fr * w); _text(img, f"{fr*tmax:.2f}", (px - 12, y0 + h + 16), 0.4)
        py = _ymap(fr * rmax, 0, rmax, y0, h); _text(img, f"{fr*rmax:.1f}", (x0 - 34, py + 4), 0.4)
    _text(img, "time (s)", (x0 + w // 2, y0 + h + 34), 0.45)
    _text(img, "residual_pos_mm (measured)", (x0 - 36, y0 - 8), 0.45)
    for r in done:
        col = GREEN if r["correction"] == "on" else RED
        px = x0 + int(float(r["time_s"]) / tmax * w)
        py = _ymap(float(r["residual_pos_mm"]), 0, rmax, y0, h)
        cv2.circle(img, (px, py), 4, col, -1)
    cv2.circle(img, (90, 450), 5, GREEN, -1); _text(img, "correction ON", (102, 454), 0.45, GREEN)
    cv2.circle(img, (300, 450), 5, RED, -1); _text(img, "correction OFF", (312, 454), 0.45, RED)
    return save_png(img, out_path)


# ── 리포트 ───────────────────────────────────────────────────────────────────
def _g(agg, ch, lv, co, field):
    v = agg.get((ch, lv, co), {}).get(field, float("nan"))
    return v


def build_context(agg, srs, fig_paths) -> dict:
    f = lambda x: f"{x:.2f}" if isinstance(x, float) and not math.isnan(x) else "n/a"
    p = lambda x: f"{x:.0f}" if isinstance(x, float) and not math.isnan(x) else "n/a"
    p1 = lambda x: f"{x:.1f}" if isinstance(x, float) and not math.isnan(x) else "n/a"
    ctx = {
        "clear_mm": srs["clearance_mm"], "clear_deg": srs["clearance_deg"],
        "gsd": srs["gsd_mm_px"], "res": srs["align_res"],
        "n_total": int(sum(a["n"] for a in agg.values())),
        "on_res": f(_g(agg, "none", "off", "on", "meas_res")),
        "off_res": f(_g(agg, "none", "off", "off", "meas_res")),
        "on_succ": p(_g(agg, "none", "off", "on", "meas_succ")),
        "off_succ": p(_g(agg, "none", "off", "off", "meas_succ")),
        "fig1": os.path.relpath(fig_paths["g1"], _HERE).replace("\\", "/"),
        "fig2": os.path.relpath(fig_paths["g2"], _HERE).replace("\\", "/"),
        "fig3": os.path.relpath(fig_paths["g3"], _HERE).replace("\\", "/"),
    }
    for ch in ("d1", "d2", "d3"):
        for lv in LEVELS:
            ctx[f"{ch}{lv}_meas"] = p(_g(agg, ch, lv, "on", "meas_succ"))
            ctx[f"{ch}{lv}_gt"] = p1(_g(agg, ch, lv, "on", "gt_succ"))
            ctx[f"{ch}{lv}_mres"] = f(_g(agg, ch, lv, "on", "meas_res"))
            ctx[f"{ch}{lv}_gtres"] = f(_g(agg, ch, lv, "on", "gt_res"))
            ctx[f"{ch}{lv}_time"] = f(_g(agg, ch, lv, "on", "time"))
    return ctx


def build_report_md(c: dict) -> str:
    return f"""# 광학 부품 정밀 정렬 — 외란 강건성 리포트

MuJoCo + Franka Emika Panda 시뮬레이션에서 외부 비전 모듈(JSON 연동)로 광학 부품을
정밀 정렬하고, 현실 외란(조명·노이즈·좌표 드리프트)에 대한 보정 루프의 강건성을 측정했다.
모든 수치는 `experiments/results.csv`({c['n_total']} trial)의 조건별 집계에서 산출했다.
SRS 허용오차: 위치 ±{c['clear_mm']}mm / 각도 ±{c['clear_deg']}deg, GSD {c['gsd']} mm/px (top 카메라 {c['res']}).

## 1. 프로그램 구조 (Flowchart)

```mermaid
flowchart TD
    A[boot.py: 모델 로드 / home / top 카메라 / 렌더러] --> B[scene_parts.py: ArUco 부품 mocap 추가]
    B --> C[run_cycle.py: PICK -> TRANSPORT -> ALIGN -> PLACE]
    C --> D{{ALIGN 보정 루프}}
    D -->|렌더 RGB+depth| E[disturbance.py: D1 조명 / D2 노이즈 / D3 드리프트]
    E --> F[interface.py: JSON 요청/응답]
    F --> G[m1_pose.py: ArUco 검출 -> pose]
    G --> H[m2_align.py: 편차 / 보정벡터]
    H --> D
    D -->|within ±{c['clear_mm']}mm/±{c['clear_deg']}deg 또는 상한10| I[run_experiment.py: KPI CSV 누적]
    I --> J[make_figures.py: CSV 집계 -> 그래프 + 리포트]
```
파이프라인: 부팅(S1) → JSON 배관(S2) → 비전 M1/M2(S3) → 보정 폐루프(S4) → 외란 주입(S5)
→ 실험·로깅(S6) → 리포트(S7). 시뮬(/sim)과 외부 비전 모듈(/external_module)은 파일·프로세스로 분리.

## 2. 로봇 및 End-Effector

- 로봇: **Franka Emika Panda** (MuJoCo Menagerie 공식 모델, 무수정 로드). 팔 7축(actuator1~7) + 그리퍼.
- End-effector: 2지 평행 그리퍼. 액추에이터 `actuator8`(tendon 'split'), ctrlrange 0~255(0=닫힘/255=열림),
  손가락 관절 `finger_joint1/2`(열림 간격 ≈0.08m). 'home' 키프레임으로 초기 자세 유지(중력 붕괴 방지).
- 자유도 nq=9, 액추에이터 nu=8. 카메라는 씬에 없어 코드측 top-down free camera(elevation=-90)로 구성.
- 본 단계 범위에서 부품은 mocap으로 추상화(팔 home 고정)하여 정렬 보정 루프의 정확도·종료 로직에 집중.

## 3. 작업 절차

1. **집기(PICK)**: home 자세에서 그리퍼 폐합(집기 추상화).
2. **이송(TRANSPORT)**: 부품을 source→목표 근처(배치오차 포함)로 단순 웨이포인트 이동.
3. **측정·정렬(ALIGN, 폐루프)**: top 카메라 렌더({c['res']}) → 외부 비전 M1 검출 → M2 편차/보정
   → 부품 보정 적용 → 재측정. **허용오차 ±{c['clear_mm']}mm/±{c['clear_deg']}deg 진입 또는 보정 상한 10회**에서 종료.
4. **배치(PLACE)**: 그리퍼 개방. 검출 실패 시 사이클 ABORT(크래시 없이 사유 기록).
   보정 전/후 오차를 측정·실제(ground-truth) 둘 다 기록.

## 4. 외부 비전 모듈 (이유·동작·연동 인터페이스)

- **분리 이유**: 비전(인지)을 시뮬 물리와 분리해 ① 독립 검증/교체 가능, ② 실제 카메라 모듈로의
  교체 경로 확보, ③ JSON 계약으로 sim↔perception 결합도 최소화. ROS2 없이 경량 JSON 통신.
- **동작**: 이미지 수신 → ArUco(DICT_4X4_50) 1차 검출, 실패 시 컨투어(형상 게이트) 보조,
  둘 다 실패 시 `DetectionError`. 픽셀→평면(m)은 2점 캘리브 선형매핑(GSD {c['gsd']} mm/px).
  M2가 `error=target-estimated`, `correction=gain·error`(부품을 목표로 이동) 산출.
- **연동 인터페이스(JSON 계약)**:
  - 요청 `{{type, image(b64 PNG), depth(b64 npy), target_pose{{x,y,theta}}, cycle}}`
  - 응답 `{{estimated_pose, error{{dx,dy,dtheta}}, correction{{dx,dy,dtheta}}, within_tolerance}}`
  - 단위: x,y=m / theta=rad, 기준=top 카메라 평면. 인프로세스 호출 + `handle_json(str)->str` 경계로
    소켓/서브프로세스 전환 대비.

## 5. 결과 및 한계

### 5.1 보정의 가치 (보정 ON vs OFF, baseline)
- 보정 OFF(대조군): 성공률 {c['off_succ']}%, 측정 잔류 {c['off_res']}mm.
- 보정 ON(외란 없음): 성공률 {c['on_succ']}%, 측정 잔류 {c['on_res']}mm.
- → 외부 비전 보정이 **{c['off_res']}mm 오차를 {c['on_res']}mm로 흡수**, 성공률 {c['off_succ']}%→{c['on_succ']}%. 모듈의 핵심 가치.

![G2 Correction ON/OFF]({c['fig2']})

### 5.2 외란별 성격 (G1, 보정 ON)
- **D1 조명·반사**: weak/med/strong 측정성공 {c['d1weak_meas']}/{c['d1med_meas']}/{c['d1strong_meas']}%,
  실제성공 {c['d1weak_gt']}/{c['d1med_gt']}/{c['d1strong_gt']}%. 실제 잔류 strong에서 {c['d1strong_gtres']}mm —
  **조명도 강하면 실제 정렬이 틀어질 수 있음**(측정과 실제의 괴리 발생).
- **D2 노이즈**: 측정성공 {c['d2weak_meas']}/{c['d2med_meas']}/{c['d2strong_meas']}%,
  실제성공 {c['d2weak_gt']}/{c['d2med_gt']}/{c['d2strong_gt']}%. 수렴 시간 {c['d2weak_time']}→{c['d2med_time']}→{c['d2strong_time']}s(강할수록 느려짐).
- **D3 좌표 드리프트**: 측정성공 {c['d3weak_meas']}/{c['d3med_meas']}/{c['d3strong_meas']}%,
  실제성공 {c['d3weak_gt']}/{c['d3med_gt']}/{c['d3strong_gt']}%. 측정상 수렴하나 실제 오차가 크게 남음(false-convergence 주범).

![G1 Disturbance vs Performance]({c['fig1']})

### 5.3 정확도-속도 트레이드오프 (G3)
- 외란이 강할수록 보정 반복(iters)·시간 증가. 보정 ON 점들은 낮은 측정 잔류에 모이고 OFF 점들은 큰 잔류에 분포.

![G3 Accuracy-Speed]({c['fig3']})

### 5.4 한계 (Limitations)
- **False-convergence (핵심 신뢰성 리스크)**: D3 strong에서 **측정 성공률 {c['d3strong_meas']}% vs 실제 성공률 {c['d3strong_gt']}%**.
  측정 잔류는 {c['d3strong_mres']}mm로 작으나 실제 잔류는 **{c['d3strong_gtres']}mm** — 시스템이 "정렬 완료"로 보고해도
  실제로는 크게 틀어져 있다. 비전이 자기 측정만 신뢰하면 오차를 "성공"으로 오인 → 외부 절대기준 교차검증 필요.
  (D2 strong {c['d2strong_meas']}%/{c['d2strong_gt']}%, D1 strong {c['d1strong_meas']}%/{c['d1strong_gt']}% 에서도 동일 괴리.)
- **D3 ground-truth 잔류오차의 이산 양자화**: 드리프트를 cycle당 고정 스텝으로 모델링해 gt 오차가 이산적으로 나타난다.
  실제 열팽창은 연속적이므로 **연속 드리프트 모델**로의 확장이 향후 과제.
- **Sim-to-Real 갭**: 핸드아이 캘리브레이션, 실제 센서 노이즈/롤링셔터, 실제 표면 반사·재질 미반영.
  부품은 mocap 추상화(물리 파지·접촉 동역학 미구현), 정렬은 검출 가능 영역(|y|≥0.4)에 한정.

### 5.5 요약 KPI (CSV 집계)
| 조건(보정 ON) | 측정 성공률 | 실제 성공률 | 측정 잔류 | 실제 잔류 |
|------|-----------|-----------|---------|---------|
| baseline(외란없음) | {c['on_succ']}% | — | {c['on_res']}mm | — |
| D1 strong | {c['d1strong_meas']}% | {c['d1strong_gt']}% | {c['d1strong_mres']}mm | {c['d1strong_gtres']}mm |
| D2 strong | {c['d2strong_meas']}% | {c['d2strong_gt']}% | {c['d2strong_mres']}mm | {c['d2strong_gtres']}mm |
| D3 strong | {c['d3strong_meas']}% | {c['d3strong_gt']}% | {c['d3strong_mres']}mm | {c['d3strong_gtres']}mm |
| baseline 보정 OFF | {c['off_succ']}% | — | {c['off_res']}mm | — |
"""


def fill_report(draft_path: str, out_path: str, context: dict) -> str:
    if os.path.isfile(draft_path):
        text = open(draft_path, "r", encoding="utf-8").read()
        for k, v in context.items():
            text = text.replace("{{" + k + "}}", str(v))
    else:
        text = build_report_md(context)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def write_summary_csv(agg: dict, path: str) -> None:
    order = {"off": 0, "weak": 1, "med": 2, "strong": 3}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "level", "correction", "n", "meas_succ_%", "gt_succ_%",
                    "mean_residual_mm", "mean_gt_residual_mm", "mean_iters", "mean_time_s", "abort"])
        for key in sorted(agg, key=lambda k: (k[0], order.get(k[1], 9), k[2])):
            a = agg[key]
            w.writerow([key[0], key[1], key[2], a["n"], f"{a['meas_succ']:.1f}", f"{a['gt_succ']:.1f}",
                        f"{a['meas_res']:.3f}", f"{a['gt_res']:.3f}", f"{a['iters']:.2f}",
                        f"{a['time']:.3f}", a["abort"]])


def check_required_sections(report_path: str, required: list) -> list:
    text = open(report_path, "r", encoding="utf-8").read()
    return [r for r in required if r not in text]


def main() -> int:
    try:
        import sys
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    rows = load_csv(RESULTS_CSV)
    print(f"[CSV] {RESULTS_CSV} 로드: {len(rows)} trial")
    agg = aggregate(rows)
    print_table(agg)
    write_summary_csv(agg, os.path.join(_HERE, "..", "experiments", "summary.csv"))

    fig_paths = {
        "g1": fig_disturbance_curve(agg, os.path.join(FIG_DIR, "disturbance_curve.png")),
        "g2": fig_correction_compare(agg, os.path.join(FIG_DIR, "correction_compare.png")),
        "g3": fig_tradeoff(rows, os.path.join(FIG_DIR, "tradeoff.png")),
    }
    for name, p in fig_paths.items():
        assert os.path.isfile(p) and os.path.getsize(p) > 0, f"그래프 미생성: {name}"
        print(f"[그래프] {name}: {p} ({os.path.getsize(p)} B)")

    ctx = build_context(agg, SRS, fig_paths)
    report_path = fill_report(os.path.join(_HERE, "report_draft_v2.md"),
                              os.path.join(_HERE, "robustness_report.md"), ctx)
    print(f"[리포트] {report_path}")

    required = ["1. 프로그램 구조", "2. 로봇 및 End-Effector", "3. 작업 절차",
               "4. 외부 비전 모듈", "5. 결과 및 한계"]
    missing = check_required_sections(report_path, required)
    assert not missing, f"필수 항목 누락: {missing}"
    txt = open(report_path, "r", encoding="utf-8").read()
    for fn in ("disturbance_curve.png", "correction_compare.png", "tradeoff.png"):
        assert fn in txt, f"그래프 임베드 누락: {fn}"
    placeholder = re.compile(r"\{\{[^}]+\}\}|\[측정값\]|\[X\]|\[Y\]|\[TBD\]|\[TODO\]")
    left = placeholder.findall(txt)
    assert not left, f"미치환 자리표시자 잔존: {left}"
    print("[검증] 5항목 존재 / 그래프 임베드 / 자리표시자 0개 OK")

    print("S7 REPORT: PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
