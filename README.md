# MuJoCo Franka Panda — 외란 강건성 정밀 정렬 (외부 비전 보정 루프)

MuJoCo + Franka Emika Panda 시뮬에서 광학 부품(ArUco 마커)을 외부 비전 모듈(JSON 연동)로
정밀 정렬하고, 현실 외란(조명·노이즈·좌표 드리프트)에 대한 보정 루프의 강건성을 측정한다.
SRS 허용오차 ±0.5mm / ±2°, GSD 1.4475 mm/px(top 카메라 1280×960).

## 설치 (requirements)

```bash
python -m venv mujoco_env && mujoco_env/Scripts/activate   # Windows
pip install mujoco==3.10.0 numpy==2.5.0 opencv-python==4.13.0   # (sim/requirements.txt 와 동일)
# 공식 모델
git clone https://github.com/google-deepmind/mujoco_menagerie
export PANDA_XML=.../mujoco_menagerie/franka_emika_panda/scene.xml   # Win: set/$env:PANDA_XML
```
- 외부 라이브러리는 `mujoco`, `numpy`, `opencv-python`(+표준 라이브러리)로 한정. ROS2 미사용. matplotlib 미사용.
- 비ASCII(한글) 경로 대응: 모델은 chdir+basename 로드, 이미지/CSV는 cv2.imencode+파이썬 파일IO 저장.
- headless/검은 화면 시 `MUJOCO_GL`(Windows 기본 wgl, Linux egl/osmesa) 지정.

## 실행 순서

```bash
python sim/boot.py                       # S1 부팅: 모델 로드·렌더·그리퍼  -> S1 BOOT: PASS
python external_module/interface.py      # S2 JSON 코덱/스키마(더미)        -> S2 INTERFACE: PASS
python sim/loop.py                       # S2 폐루프 배관                   -> S2 LINK: PASS
python external_module/m2_align.py       # S3 편차/보정 산출                -> S3 M2: PASS
python external_module/scene_parts.py    # S3 부품 씬 + ArUco 검출          -> S3 SCENE: PASS
python external_module/m1_pose.py        # S3 비전 M1+M2 통합               -> S3 VISION: PASS
python sim/run_cycle.py                  # S4 보정 폐루프 1사이클           -> S4 CYCLE: PASS
python sim/disturbance.py                # S5 외란 D1/D2/D3 주입            -> S5 DISTURB: PASS
# S6 캠페인(외란×강도×보정 N회) -> results.csv 누적
python experiments/run_experiment.py --disturb none --levels off --correction both --trials 8 --overwrite
python experiments/run_experiment.py --disturb d1 --levels weak,med,strong --correction both --trials 8
python experiments/run_experiment.py --disturb d2 --levels weak,med,strong --correction both --trials 8
python experiments/run_experiment.py --disturb d3 --levels weak,med,strong --correction both --trials 8
python report/make_figures.py            # S7 CSV 집계 -> 그래프 3종 + 리포트 -> S7 REPORT: PASS
```

## 저장소 구조

```
sim/
  boot.py            # S1 모델 로드/home/카메라/렌더/그리퍼 (load_model, render_rgb/depth, actuate_gripper ...)
  loop.py            # S2 시뮬측 폐루프 배관(이미지→요청→응답→더미 보정)
  run_cycle.py       # S4 PICK→TRANSPORT→ALIGN(보정 폐루프)→PLACE 상태기계 (render_fn/apply_correction 훅)
  disturbance.py     # S5 외란 D1(조명·반사)/D2(노이즈)/D3(좌표 드리프트), 강도·토글
  requirements.txt
external_module/
  interface.py       # S2 JSON 코덱·스키마·handle_json + S4 실제 perception(configure_perception)
  m1_pose.py         # S3 ArUco 1차/컨투어 보조 검출, 픽셀↔미터 캘리브, DetectionError
  m2_align.py        # S3 편차(error)/보정(correction) 산출(S2 응답 계약 호환)
  scene_parts.py     # S3 공식 씬에 ArUco 부품을 MjSpec mocap으로 런타임 추가(XML 무작성)
  aruco_marker.png, scene_part_top.png
experiments/
  run_experiment.py  # S6 외란×강도×보정 매트릭스 N회, KPI 4종 CSV 누적, CLI
  results.csv, summary.csv, run_images/, disturbance_samples/
report/
  make_figures.py    # S7 results.csv 집계 → KPI 그래프 3종(opencv PNG) + 리포트 자동 채움
  robustness_report.md           # 최종 리포트(과제 필수 5항목)
  figures/{disturbance_curve,correction_compare,tradeoff}.png
docs/
  interface_contract.md          # JSON 계약·좌표·외란 규약
```

## 핵심 결과 (CSV 160 trial 집계)

- 보정 ON vs OFF: 잔류 9.84mm→0.38mm, 성공률 0%→100%.
- **False-convergence(핵심 신뢰성 리스크)**: 측정 성공률과 실제 성공률의 괴리.
  D3 strong 측정 100% / 실제 12.5%(실제 잔류 11.62mm), D2 strong 75%/37.5%, D1 strong 100%/0%.
  → 비전이 자기 측정만 신뢰하면 큰 오차를 "성공"으로 오인. 외부 절대기준 교차검증 필요.
- 상세: `report/robustness_report.md`, 그래프 `report/figures/`.

## 검증 환경

mujoco 3.10.0 / numpy 2.5.0 / opencv-python 4.13.0, Windows 11 / Python 3.13,
MUJOCO_GL 미설정(wgl), Menagerie 커밋 accb6df. 각 모듈 진입점에 self-test 포함(`... PASS`).
