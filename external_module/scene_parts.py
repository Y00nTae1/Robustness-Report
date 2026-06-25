#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (광학 부품 런타임 추가 — S3 비전 모듈)
================================================
공식 Franka 씬(franka_emika_panda/scene.xml)을 한 글자도 수정하지 않고 mujoco.MjSpec
으로 로드한 뒤, 코드(런타임 메모리)에서 ArUco 마커판(평평한 박스) 부품을 mocap body 로
추가한다. mocap 이라 dynamics 없이 pose 를 직접 텔레포트할 수 있어 렌더 결정성이 높다.

제약 준수 근거(커스텀 모델 미작성):
  - 디스크에 새 로봇 모델/URDF/커스텀 XML 을 저작하지 않는다.
  - 공식 모델은 무수정 로드, 추가물은 일반 프리미티브(박스)+텍스처뿐(로봇 모델링 아님).
  - S1 의 "카메라를 XML 에 넣지 않고 코드로 free camera 구성"과 동일한 성격(씬 셋업).

좌표 계약: pose={x,y,theta}, 단위 x,y=m / theta=rad, 기준=top 카메라 평면(S2 계약).
비ASCII 경로 우회: 모델 로드는 chdir+basename, 마커 PNG 저장/로드는 imencode/파일IO.

실행 방법(단독 self-test):
  set PANDA_XML=  (franka_emika_panda/scene.xml 의 경로)
  python external_module/scene_parts.py
  -> 부품 씬 compile + 상단 렌더에서 ArUco 검출 확인 후 "S3 SCENE: PASS".
"""

import os
import sys
import math

import numpy as np
import cv2
import mujoco


DEFAULT_DICT = "DICT_4X4_50"
DEFAULT_Z = 0.02


# ---------------------------------------------------------------------------
# ArUco 자산
# ---------------------------------------------------------------------------
def _aruco_dictionary(dict_name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def aruco_rgb(marker_id: int = 0, dict_name: str = DEFAULT_DICT,
              pixels: int = 480, border_frac: float = 0.25) -> np.ndarray:
    """ArUco 마커 RGB(흰 quiet zone 포함) 생성. 검출 안정성 위해 테두리 필수."""
    img = cv2.aruco.generateImageMarker(_aruco_dictionary(dict_name), marker_id, pixels)
    b = int(pixels * border_frac)
    canvas = np.full((pixels + 2 * b, pixels + 2 * b), 255, np.uint8)
    canvas[b:b + pixels, b:b + pixels] = img
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def _imwrite_unicode(path: str, bgr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise IOError(f"PNG 인코딩 실패: {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def _imread_unicode(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        buf = f.read()
    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise IOError(f"PNG 디코딩 실패: {path}")
    return arr


def generate_aruco_png(out_path: str, marker_id: int = 0,
                       dict_name: str = DEFAULT_DICT, pixels: int = 480) -> str:
    """ArUco 마커 이미지를 PNG 로 저장(텍스처 소스 자산). 경로 반환."""
    rgb = aruco_rgb(marker_id, dict_name, pixels)
    _imwrite_unicode(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return out_path


# ---------------------------------------------------------------------------
# 씬 합성
# ---------------------------------------------------------------------------
def build_scene_with_part(model_path: str, marker_png: str,
                          part_half: float = 0.11, part_xy=(0.30, 0.40),
                          marker_id: int = 0, dict_name: str = DEFAULT_DICT,
                          z: float = DEFAULT_Z, offwidth: int = 1280, offheight: int = 960):
    """MjSpec 로 공식 씬 로드 → ArUco 텍스처 박스를 mocap body 로 추가 → compile.

    offwidth/offheight: 오프스크린 프레임버퍼 크기(고해상도 정밀 측정 지원).
    반환: (model, data, part_info)
      part_info = {mocap_id, geom_id, body_id, half, z, marker_id, dict_name}
    """
    marker = cv2.cvtColor(_imread_unicode(marker_png), cv2.COLOR_BGR2RGB)
    H, W = marker.shape[:2]

    model_dir = os.path.dirname(os.path.abspath(model_path))
    model_file = os.path.basename(model_path)
    prev_cwd = os.getcwd()
    try:
        os.chdir(model_dir)  # 비ASCII 경로 우회 + 상대 include/asset 해석
        spec = mujoco.MjSpec.from_file(model_file)
        spec.visual.global_.offwidth = offwidth     # 고해상도 오프스크린 허용
        spec.visual.global_.offheight = offheight

        tex = spec.add_texture()
        tex.name = "aruco_tex"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.width, tex.height, tex.nchannel = W, H, 3
        tex.data = marker.tobytes()

        mat = spec.add_material()
        mat.name = "aruco_mat"
        mat.texrepeat = [1, 1]
        mat.texuniform = False
        mat.specular = mat.shininess = mat.reflectance = 0.0
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "aruco_tex"

        body = spec.worldbody.add_body()
        body.name = "part"
        body.mocap = True
        body.pos = [part_xy[0], part_xy[1], z]
        g = body.add_geom()
        g.name = "part_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [part_half, part_half, 0.002]
        g.material = "aruco_mat"
        g.contype = g.conaffinity = 0   # 충돌 비활성(마커판은 시각용)

        model = spec.compile()
    finally:
        os.chdir(prev_cwd)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part")
    part_info = {
        "body_id": body_id,
        "mocap_id": int(model.body_mocapid[body_id]),
        "geom_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "part_geom"),
        "half": part_half, "z": z,
        "marker_id": marker_id, "dict_name": dict_name,
    }
    return model, data, part_info


def set_part_pose(model, data, pose: dict, part_info: dict) -> None:
    """mocap_pos/mocap_quat 로 부품 (x,y,theta) 직접 세팅 후 mj_forward."""
    mid = part_info["mocap_id"]
    z = pose.get("z", part_info["z"])
    t = float(pose["theta"])
    data.mocap_pos[mid] = [float(pose["x"]), float(pose["y"]), z]
    data.mocap_quat[mid] = [math.cos(t / 2.0), 0.0, 0.0, math.sin(t / 2.0)]
    mujoco.mj_forward(model, data)


def get_part_pose(model, data, part_info: dict) -> dict:
    """부품 ground-truth pose {x,y,theta} (m·rad)."""
    mid = part_info["mocap_id"]
    px, py, _ = data.mocap_pos[mid]
    w, _, _, zq = data.mocap_quat[mid]
    theta = 2.0 * math.atan2(zq, w)
    return {"x": float(px), "y": float(py), "theta": float(math.atan2(math.sin(theta), math.cos(theta)))}


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _import_boot():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
    import boot
    return boot


def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")  # cp949 콘솔에서 비호환 글리프 크래시 방지
    except Exception:
        pass
    boot = _import_boot()
    model_path = os.environ.get(
        "PANDA_XML",
        os.path.join("mujoco_menagerie", "franka_emika_panda", "scene.xml"))

    here = os.path.dirname(os.path.abspath(__file__))
    marker_png = generate_aruco_png(os.path.join(here, "aruco_marker.png"))
    print(f"[자산] ArUco PNG 생성: {marker_png}")

    model, data, info = build_scene_with_part(model_path, marker_png)
    print(f"[씬] compile OK - nmocap={model.nmocap}, ngeom={model.ngeom}, part_info={info}")

    boot.reset_to_home(model, data)
    set_part_pose(model, data, {"x": 0.30, "y": 0.40, "theta": 0.0}, info)

    camera = boot.resolve_top_camera(model)
    renderer = boot.make_renderer(model, 640, 480)
    try:
        rgb = boot.render_rgb(model, data, camera, renderer)
        boot.save_rgb(rgb, os.path.join(here, "scene_part_top.png"))
    finally:
        renderer.close()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    det = cv2.aruco.ArucoDetector(_aruco_dictionary(DEFAULT_DICT),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    assert ids is not None and info["marker_id"] in ids.ravel().tolist(), \
        "상단 렌더에서 ArUco 마커가 검출되지 않음"
    print(f"[검증] 상단 렌더에서 ArUco 검출 OK ids={ids.ravel().tolist()}, "
          f"저장: scene_part_top.png")

    print("S3 SCENE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
