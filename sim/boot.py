#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (S1 환경 부팅 엔트리포인트)
====================================
MuJoCo Menagerie의 Franka Panda 공식 MJCF를 로드하고, home 키프레임으로 자세를
세운 뒤, 상단 시점(top-down free camera)으로 RGB + depth 를 렌더해 파일로 저장한다.
그리퍼 개폐를 명령하여 손가락 간격 변화(수치) + 열림/닫힘 RGB(시각 증거) 로 검증한다.

산출물:
  - sim/rgb_top.png        : 상단 카메라 RGB (uint8, BGR 저장)
  - sim/depth_top.png      : 깊이 가시화용 정규화 PNG (uint8)
  - sim/depth_top.npy      : 깊이 원본 배열 (float32, 미터 단위)
  - sim/gripper_open.png   : 그리퍼 열림 상태 RGB (시각 증거)
  - sim/gripper_closed.png : 그리퍼 닫힘 상태 RGB (시각 증거)

제약:
  - 외부 라이브러리는 mujoco, numpy, opencv-python(cv2) 로 한정.
  - 커스텀 URDF/모델 미작성. Menagerie 공식 모델만 경로로 참조.
  - 가시성 확보를 위해 XML 에 카메라/부품을 추가하지 않는다. 상단 뷰는 코드측
    free camera(elevation=-90) 로만 구성한다(계획 리스크 #1 회피).
  - 베이스 Franka 씬에는 광학 '부품'이 없으므로 S1 에서는 '로봇이 상단 시점에 보임'을
    가시성 기준으로 삼는다(세그멘테이션 렌더로 로봇 픽셀 비율을 정량 측정).

실행 방법:
  1) 의존성:  pip install mujoco numpy opencv-python   (검증 버전은 sim/requirements.txt)
  2) 모델:    git clone https://github.com/google-deepmind/mujoco_menagerie
  3) 경로:    PANDA_XML=.../franka_emika_panda/scene.xml  또는  --model 인자
  4) 실행:    python sim/boot.py
  5) headless/검은 화면 시 렌더 백엔드 지정:
                Linux  : export MUJOCO_GL=egl   (또는 osmesa)
                Windows: 기본(wgl) 사용. 문제 시 set MUJOCO_GL=glfw
  6) self-test: main() 이 곧 self-test. 모든 assert 통과 시 "S1 BOOT: PASS", 종료코드 0.

import 시에는 아무 동작도 실행하지 않는다(함수 정의만). 단독 실행 시에만 main() 수행.
"""

import os
import sys
import argparse

import numpy as np
import mujoco
import cv2


# ---------------------------------------------------------------------------
# 1. 모델 로딩 & 환경 검증
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    """MJCF 경로를 받아 (MjModel, MjData) 를 반환한다."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {model_path}\n"
            "PANDA_XML 환경변수나 --model 인자로 Menagerie 의 "
            "franka_emika_panda/scene.xml 경로를 지정하세요."
        )
    # 한글/유니코드 경로(예: C:\Users\태원\...)는 MuJoCo C++ fopen 에서 깨진다.
    # 모델 디렉터리로 chdir 후 ASCII 파일명만 넘겨 유니코드 바이트 전달을 피한다.
    model_dir = os.path.dirname(os.path.abspath(model_path))
    model_file = os.path.basename(model_path)
    prev_cwd = os.getcwd()
    try:
        os.chdir(model_dir)
        model = mujoco.MjModel.from_xml_path(model_file)
    finally:
        os.chdir(prev_cwd)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _names(model, objtype):
    """모델 내 특정 객체 타입의 (id, name) 목록."""
    n = {
        mujoco.mjtObj.mjOBJ_CAMERA: model.ncam,
        mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
        mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
    }[objtype]
    return [(i, mujoco.mj_id2name(model, objtype, i)) for i in range(n)]


def describe_model(model) -> None:
    """카메라/액추에이터/관절 이름 목록을 출력해 환경을 검증한다."""
    print("[검증] 카메라    :", _names(model, mujoco.mjtObj.mjOBJ_CAMERA) or "(없음)")
    print("[검증] 액추에이터:", _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR))
    fingers = [nm for _, nm in _names(model, mujoco.mjtObj.mjOBJ_JOINT)
               if nm and "finger" in nm.lower()]
    print("[검증] 손가락 관절:", fingers or "(이름에 'finger' 없음)")


def reset_to_home(model, data) -> None:
    """home 키프레임이 있으면 그 자세/ctrl 로 리셋(팔 hold), 없으면 현재 qpos hold.

    이렇게 해야 팔 액추에이터 ctrl 이 0 으로 남아 중력에 붕괴하지 않는다.
    """
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)  # qpos/qvel/ctrl 까지 설정
    else:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        # 폴백: 관절 구동 액추에이터를 현재 qpos 에 맞춰 hold
        for aid in range(model.nu):
            if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = model.actuator_trnid[aid, 0]
                data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# 2. 상단 카메라 설정 해결
# ---------------------------------------------------------------------------
def resolve_top_camera(model, preferred_name: str = "top"):
    """상단 시점 카메라를 결정한다.

    - 모델에 preferred_name(또는 이름에 'top' 포함) 카메라가 있으면 그 id(int).
    - 없으면 모델 stat(center/extent)에 맞춰 top-down free camera 를 구성해 반환.
    """
    for cam_id, name in _names(model, mujoco.mjtObj.mjOBJ_CAMERA):
        if name and (name == preferred_name or "top" in name.lower()):
            return cam_id

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)  # lookat=stat.center, distance≈1.5*extent
    cam.elevation = -90.0   # 정수직 하향
    cam.azimuth = 90.0
    # 로봇이 프레임을 적절히 채우도록 약간 당긴다(top-down 에서 너무 멀어지는 것 방지).
    cam.distance = float(model.stat.extent) * 1.3
    return cam


# ---------------------------------------------------------------------------
# 3 & 4. 렌더링 (단일 Renderer 재사용으로 GL 컨텍스트 churn 제거)
# ---------------------------------------------------------------------------
def make_renderer(model, width: int = 640, height: int = 480):
    """오프스크린 Renderer 생성. GL 컨텍스트 실패 시 안내 메시지를 덧붙인다."""
    try:
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "오프스크린 렌더러 생성 실패. headless/원격이면 MUJOCO_GL 환경변수를 "
            "설정하세요 (Linux: 'egl'/'osmesa', Windows: 'glfw').\n"
            f"원본 오류: {exc!r}"
        ) from exc


def render_rgb(model, data, camera, renderer) -> np.ndarray:
    """RGB 프레임. 반환: uint8 (H, W, 3)."""
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera)
    return renderer.render().copy()


def render_depth(model, data, camera, renderer) -> np.ndarray:
    """깊이 맵. 반환: float32 (H, W), 미터 단위."""
    mujoco.mj_forward(model, data)
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data, camera)
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    return depth.astype(np.float32, copy=False)


def robot_pixel_ratio(model, data, camera, renderer) -> float:
    """세그멘테이션 렌더로 '로봇 geom 픽셀 / 전체 픽셀' 비율을 측정한다.

    바닥(floor) geom 과 배경(skybox, id=-1)을 제외한 geom 픽셀 비율 → 로봇 가시성.
    """
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mujoco.mj_forward(model, data)
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(data, camera)
        seg = renderer.render()  # (H, W, 2) int32: [...,0]=objid, [...,1]=objtype
    finally:
        renderer.disable_segmentation_rendering()
    obj_id = seg[..., 0]
    obj_type = seg[..., 1]
    is_geom = obj_type == int(mujoco.mjtObj.mjOBJ_GEOM)
    robot = is_geom & (obj_id >= 0) & (obj_id != floor_id)
    return float(robot.mean())


# ---------------------------------------------------------------------------
# 5. 이미지 저장
# ---------------------------------------------------------------------------
def _imwrite_unicode(path: str, img: np.ndarray) -> None:
    """유니코드 경로 안전 저장: cv2.imencode 후 파이썬 파일 IO 로 기록.

    (cv2.imwrite 는 Windows 에서 한글 등 비ASCII 경로를 제대로 쓰지 못한다.)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"이미지 인코딩 실패: {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def save_rgb(rgb: np.ndarray, out_path: str) -> None:
    """RGB(uint8, H,W,3) 를 BGR 로 변환해 PNG 저장."""
    _imwrite_unicode(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_depth(depth: np.ndarray, png_path: str, npy_path: str) -> None:
    """깊이를 가시화 PNG(정규화 uint8) + 원본 .npy(float32) 로 저장."""
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    np.save(npy_path, depth.astype(np.float32, copy=False))

    finite = np.isfinite(depth)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        d = depth[finite]
        dmin, dmax = float(d.min()), float(d.max())
        norm = (depth - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(depth)
        norm = np.clip(norm, 0.0, 1.0)
        vis = ((1.0 - norm) * 255.0).astype(np.uint8)  # 가까울수록 밝게
        vis[~finite] = 0
    _imwrite_unicode(png_path, vis)


# ---------------------------------------------------------------------------
# 6. 그리퍼 개폐 확인
# ---------------------------------------------------------------------------
def _finger_joint_addrs(model):
    """이름에 'finger' 가 포함된 관절들의 qpos 주소 목록."""
    addrs = [model.jnt_qposadr[jid]
             for jid, name in _names(model, mujoco.mjtObj.mjOBJ_JOINT)
             if name and "finger" in name.lower()]
    if not addrs:
        raise RuntimeError("손가락 관절(이름에 'finger' 포함)을 찾지 못했습니다.")
    return addrs


def _gripper_actuator_id(model):
    """그리퍼 액추에이터 id 를 이름 휴리스틱으로 찾고, 실패 시 마지막 액추에이터."""
    for aid, name in _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR):
        if name and any(k in name.lower() for k in ("finger", "hand", "grip")):
            return aid
    # Franka(Menagerie): 팔 7축 + 그리퍼 1(actuator8 = tendon 'split'). 마지막이 그리퍼.
    if model.nu == 0:
        raise RuntimeError("액추에이터가 없습니다.")
    return model.nu - 1


def _finger_gap(model, data, addrs) -> float:
    """손가락 관절 qpos 합 = 그리퍼 개도(간격, m)."""
    return float(sum(data.qpos[a] for a in addrs))


def actuate_gripper(model, data, opening: float, settle_steps: int = 1500,
                    tol: float = 1e-4) -> float:
    """그리퍼를 opening(0=닫힘 ~ 1=열림) 비율로 명령하고 간격이 수렴할 때까지 step.

    opening 은 그리퍼 액추에이터 ctrlrange 로 선형 매핑된다(Franka: 0→닫힘, 255→열림).
    settle_steps 는 하드 상한, tol 은 연속 측정 간 수렴 임계.
    출력 : 수렴 후 손가락 간격(m)
    """
    opening = float(np.clip(opening, 0.0, 1.0))
    aid = _gripper_actuator_id(model)
    lo, hi = model.actuator_ctrlrange[aid]
    if not (hi > lo):
        lo, hi = 0.0, 1.0
    data.ctrl[aid] = lo + opening * (hi - lo)

    addrs = _finger_joint_addrs(model)
    prev = _finger_gap(model, data, addrs)
    steps = 0
    batch = 10
    while steps < settle_steps:
        for _ in range(batch):
            mujoco.mj_step(model, data)
        steps += batch
        cur = _finger_gap(model, data, addrs)
        if abs(cur - prev) < tol:
            break
        prev = cur
    return _finger_gap(model, data, addrs)


def verify_gripper(model, data, camera, renderer, outdir, threshold: float = 0.01):
    """열림/닫힘을 명령해 간격 변화(수치)와 RGB(시각 증거)로 검증한다.

    - home 으로 리셋 후 열림→간격/이미지, 다시 리셋 후 닫힘→간격/이미지.
    - 판정: |Δ|>threshold AND gap_open>gap_closed (방향까지 엄격 확인).
    출력 : (ok: bool, gap_open: float, gap_closed: float, open_png, closed_png)
    """
    open_png = os.path.join(outdir, "gripper_open.png")
    closed_png = os.path.join(outdir, "gripper_closed.png")

    reset_to_home(model, data)
    gap_open = actuate_gripper(model, data, opening=1.0)
    save_rgb(render_rgb(model, data, camera, renderer), open_png)

    reset_to_home(model, data)
    gap_closed = actuate_gripper(model, data, opening=0.0)
    save_rgb(render_rgb(model, data, camera, renderer), closed_png)

    diff = abs(gap_open - gap_closed)
    ok = (diff > threshold) and (gap_open > gap_closed)
    print(f"[그리퍼] open_gap={gap_open:.4f}m, closed_gap={gap_closed:.4f}m, "
          f"Δ={gap_open - gap_closed:+.4f}m (threshold={threshold}m) → "
          f"{'OK' if ok else 'FAIL'}")
    return ok, gap_open, gap_closed, open_png, closed_png


# ---------------------------------------------------------------------------
# 진입점 = self-test
# ---------------------------------------------------------------------------
def _default_model_path() -> str:
    return os.environ.get(
        "PANDA_XML",
        os.path.join("mujoco_menagerie", "franka_emika_panda", "scene.xml"),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S1 환경 부팅 + self-test")
    parser.add_argument("--model", default=_default_model_path(),
                        help="Menagerie Franka MJCF 경로 (기본: $PANDA_XML)")
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="산출물 저장 디렉터리 (기본: 이 파일이 있는 sim/)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-robot-ratio", type=float, default=0.01,
                        help="상단 뷰에서 로봇 픽셀이 차지해야 할 최소 비율")
    args = parser.parse_args(argv)

    rgb_path = os.path.join(args.outdir, "rgb_top.png")
    depth_png = os.path.join(args.outdir, "depth_top.png")
    depth_npy = os.path.join(args.outdir, "depth_top.npy")

    # 1) 로드 & 검증 & home 자세
    model, data = load_model(args.model)
    describe_model(model)
    reset_to_home(model, data)

    # 2) 상단 카메라
    camera = resolve_top_camera(model)
    print("[카메라] 사용 대상:",
          f"named id={camera}" if isinstance(camera, int) else "free top-down camera")

    renderer = make_renderer(model, args.width, args.height)
    try:
        # 3) RGB + 가시성(세그멘테이션) 정량 검증
        rgb = render_rgb(model, data, camera, renderer)
        assert rgb.shape == (args.height, args.width, 3), f"RGB shape 오류: {rgb.shape}"
        assert rgb.dtype == np.uint8, f"RGB dtype 오류: {rgb.dtype}"
        ratio = robot_pixel_ratio(model, data, camera, renderer)
        print(f"[가시성] 로봇 픽셀 비율 = {ratio*100:.2f}% (최소 {args.min_robot_ratio*100:.2f}%)")
        assert ratio >= args.min_robot_ratio, (
            f"로봇이 프레임에 충분히 보이지 않음(ratio={ratio:.4f}). 카메라 프레이밍 확인 필요")

        # 4) Depth
        depth = render_depth(model, data, camera, renderer)
        assert depth.shape == (args.height, args.width), f"Depth shape 오류: {depth.shape}"
        assert depth.dtype == np.float32, f"Depth dtype 오류: {depth.dtype}"
        assert np.isfinite(depth).any(), "Depth 에 유효 값이 없음"

        # 5) 저장 + 재로드 확인
        save_rgb(rgb, rgb_path)
        save_depth(depth, depth_png, depth_npy)
        assert np.load(depth_npy).shape == depth.shape, "재로드 depth shape 불일치"
        print(f"[저장] {rgb_path}\n       {depth_png}\n       {depth_npy}")

        # 6) 그리퍼 개폐 (수치 + 시각 증거)
        ok, _, _, open_png, closed_png = verify_gripper(
            model, data, camera, renderer, args.outdir)
        assert ok, "그리퍼 개폐 검증 실패"
        print(f"[저장] {open_png}\n       {closed_png}")
    finally:
        renderer.close()

    # 산출물 존재 최종 확인
    for p in (rgb_path, depth_png, depth_npy,
              os.path.join(args.outdir, "gripper_open.png"),
              os.path.join(args.outdir, "gripper_closed.png")):
        assert os.path.isfile(p), f"산출물 미생성: {p}"

    print("S1 BOOT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
