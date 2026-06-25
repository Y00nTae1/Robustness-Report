#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 책임 (시뮬 측 폐루프 배관 — S2 연동 배관)
=============================================
S1 boot.py 의 모델 로드·렌더 함수를 재사용해 매 사이클 카메라 프레임(RGB+depth)을
얻고, JSON 요청으로 만들어 외부 모듈 external_module/interface.process_request 에
보낸다. 응답의 correction 을 추적 pose 에 더하는 '더미 이동'을 적용하고,
within_tolerance 또는 max_cycles 도달 시 종료한다(실제 비전·IK 없음).

의존: sim/boot.py (동일 폴더), external_module/interface.py (코덱·인터페이스).
제약: mujoco/numpy/opencv-python + 표준 라이브러리(json/os/sys)만. ROS2 미사용.
      비ASCII 경로 우회는 boot.py 패턴(chdir+basename 로드, imencode 저장)을 그대로 승계.

실행 방법:
  set PANDA_XML=  (franka_emika_panda/scene.xml 의 경로)   또는 --model 인자
  python sim/loop.py
  → 사이클별 JSON 로그(error_norm 감소) 출력 후 "S2 LINK: PASS", 종료코드 0.

import 시에는 아무 동작도 실행하지 않는다. 단독 실행 시에만 main() 수행.
"""

import os
import sys
import json
import argparse

# 동일 폴더의 boot.py 와 ../external_module 의 interface.py 를 import 경로에 추가
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "external_module"))

import boot          # noqa: E402  (S1 산출물 재사용)
import interface     # noqa: E402  (외부 모듈)


def build_request(rgb, depth, target_pose: dict, cycle: int) -> dict:
    """렌더 프레임 + 목표 pose 로 perception_request 를 만든다."""
    return {
        "type": "perception_request",
        "image": interface.encode_image(rgb),
        "depth": interface.encode_depth(depth),
        "target_pose": target_pose,
        "cycle": cycle,
    }


def apply_correction(current_pose: dict, correction: dict) -> dict:
    """더미 로봇 이동: 실제 IK 없이 추적 pose 에 correction 을 더한다."""
    return {
        "x": current_pose["x"] + correction["dx"],
        "y": current_pose["y"] + correction["dy"],
        "theta": current_pose["theta"] + correction["dtheta"],
    }


def _error_norm(error: dict) -> float:
    return (error["dx"] ** 2 + error["dy"] ** 2 + error["dtheta"] ** 2) ** 0.5


def run_loop(model, data, camera, renderer, target_pose: dict, init_pose: dict,
             max_cycles: int = 20, tolerance: float = 1e-3) -> list:
    """렌더→요청→process_request→보정→종료판정 반복. 사이클 history 반환."""
    history = []
    current = dict(init_pose)
    reason = "max_cycles"
    for cycle in range(max_cycles):
        rgb = boot.render_rgb(model, data, camera, renderer)
        depth = boot.render_depth(model, data, camera, renderer)
        request = build_request(rgb, depth, target_pose, cycle)
        req_bytes = len(json.dumps(request))

        response = interface.process_request(request)
        mag = _error_norm(response["error"])
        current = apply_correction(current, response["correction"])

        record = {
            "cycle": cycle,
            "error_norm": round(mag, 6),
            "within_tolerance": response["within_tolerance"],
            "estimated_pose": {k: round(v, 5) for k, v in response["estimated_pose"].items()},
            "current_pose": {k: round(v, 5) for k, v in current.items()},
            "req_bytes": req_bytes,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if response["within_tolerance"] or mag < tolerance:
            reason = "within_tolerance"
            break

    print(f"[루프 종료] 사유={reason}, 총 {len(history)} 사이클")
    return history


def _default_target_init():
    # 목표는 top 카메라 평면상의 임의 좌표(m, rad). 추적 pose 는 목표와 동일 지점에서
    # 시작하되, 더미 인지가 cycle 기반 잔차를 보고하므로 correction 으로 수렴 과정을 보인다.
    target = {"x": 0.30, "y": 0.00, "theta": 0.00}
    init = {"x": 0.30, "y": 0.00, "theta": 0.00}
    return target, init


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S2 연동 배관 폐루프 + self-test")
    parser.add_argument("--model", default=boot._default_model_path(),
                        help="Menagerie Franka MJCF 경로 (기본: $PANDA_XML)")
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args(argv)

    # S1 재사용: 로드 → home 자세 → 상단 카메라 → 렌더러
    model, data = boot.load_model(args.model)
    boot.reset_to_home(model, data)
    camera = boot.resolve_top_camera(model)
    renderer = boot.make_renderer(model)
    try:
        target, init = _default_target_init()
        history = run_loop(model, data, camera, renderer, target, init,
                           max_cycles=args.max_cycles, tolerance=args.tolerance)
    finally:
        renderer.close()

    # self-test 단언
    assert history, "history 가 비어있음"
    assert history[-1]["within_tolerance"], "최종 사이클이 within_tolerance 에 도달하지 못함"
    norms = [h["error_norm"] for h in history]
    assert all(norms[i + 1] < norms[i] for i in range(len(norms) - 1)), \
        "error_norm 이 cycle 증가에 따라 단조 감소하지 않음"
    print(f"[검증] error_norm 단조감소 OK ({norms[0]:.6f} → {norms[-1]:.6f}), "
          f"{len(history)} 사이클 내 수렴")

    print("S2 LINK: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
