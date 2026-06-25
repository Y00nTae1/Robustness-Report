# 시뮬 ↔ 외부 비전 모듈 JSON 인터페이스 계약 (S2)

## 좌표 계약
- `pose = {x, y, theta}` — **단위: x,y = meter, theta = radian**
- **기준 프레임: top 카메라 이미지 평면에 투영된 좌표** (free top-down camera 기준).
- S2(현재)는 더미 단계라 값 자체는 임의지만, **키·단위·기준은 고정**한다.
  실제 부품 좌표는 S3(광학 부품 도입)에서 채운다.

## 메시지 스키마
요청 (sim → external):
```json
{
  "type": "perception_request",
  "image": "<base64 PNG (무손실)>",
  "depth": "<base64 .npy float32>",
  "target_pose": {"x": 0.30, "y": 0.0, "theta": 0.0},
  "cycle": 0
}
```
응답 (external → sim):
```json
{
  "estimated_pose": {"x": ..., "y": ..., "theta": ...},
  "error":      {"dx": ..., "dy": ..., "dtheta": ...},
  "correction": {"dx": ..., "dy": ..., "dtheta": ...},
  "within_tolerance": false
}
```
- `error = target_pose - estimated_pose`
- `correction = gain * error`
- `within_tolerance = ‖error‖ < tolerance` (기본 tolerance = 1e-3)

## 직렬화 규약
- 이미지/Depth 는 메모리 버퍼(`cv2.imencode` / `np.save(BytesIO)`)로 base64 인코딩 →
  **비ASCII(한글) 경로와 무관**. (파일 저장이 필요하면 boot.py 의 imencode+파일IO 패턴 사용)
- depth 는 float32 원본 보존(무손실). 요청 바이트 크기는 루프가 사이클마다 로깅한다.

## 전송 경계
- S2 기본: **인프로세스 함수 호출** `interface.process_request(dict) -> dict`.
- 문자열 경계 `interface.handle_json(str) -> str` 제공 → 후속 단계에서 socket/subprocess 로
  로직 변경 없이 전환 가능. **ROS2 미사용.**

## 채널·검출 규약 (S3 추가)
- 이미지 채널: 시뮬 렌더 출력은 **RGB**. M1(`m1_pose`)은 내부에서 RGB→Gray 로 명시 변환 후
  ArUco/컨투어를 수행한다(S2 인계 #3 해소).
- 픽셀↔미터: top 카메라 평면에서 **선형 매핑** `x=(u-u0)·s, y=-(v-v0)·s`.
  검증 환경 실측 `s ≈ 0.002893 m/px`, `u0≈215.8, v0≈239.6` (640×480, free top-down).
  값은 카메라/해상도에 의존 → `m1_pose.calibrate_scale`로 런타임 산출(2~3점).
- 부품 검출: ArUco(DICT_4X4_50) 1차, 컨투어(minAreaRect) 보조, 둘 다 실패 시 `DetectionError`.
- 보정 방향(S2 인계 #1): `correction=gain·(target-estimated)`를 부품 pose 에 가산하면 수렴
  (실측 닫힌 루프 err_norm 0.4187→0.0251, tol=0.03). 검출 노이즈 바닥(각도 ~0.016rad) 고려.

## 외란 모델 (S5, sim/disturbance.py)
비전 측정 계층에만 주입(부품 mocap 추상화 유지). 강도 off/weak/med/strong, 시드로 결정적.
- D1 조명·반사: `model.vis.headlight` diffuse/ambient ×{1.0,1.3,1.7,2.2}(렌더 후 원복) + 반사 글레어 spot 강도 {0,0.15,0.30,0.5}.
- D2 카메라 노이즈: 가우시안 σ(uint8) {0,5,15,30}, 0~255 클립.
- D3 좌표 드리프트: 카메라 lookat 에 cycle 비례 오프셋 {0,0.5,1.5,3.0} mm/cycle (x, y는 ×0.5). 고정 calib 대비 측정 bias = 드리프트의 정의.
- 토글: `make_config(d1,d2,d3,seed)`, off 조합은 baseline 과 비트 동일(항등성). 통합 래퍼 `disturbed_render(...,cycle)` 적용 순서 = D3(카메라)→D1(조명/글레어)→D2(노이즈).
- 실측(640×480): D2 std≈σ(4.91/14.65/28.85), D3 cycle=10 중심이동 2.24/5.59/11.63px, D1+D2(med)에서 M1 측정 0.723mm 변화. 강 레벨은 검출 실패 가능(강건성 경계, S6 통계 입력).

## 성공 기준
- `interface.py` 단독: 코덱 round-trip + 스키마 + (cycle↑ → error↓) → `S2 INTERFACE: PASS`
- `loop.py`: N사이클 내 `within_tolerance` 도달 + error_norm 단조감소 → `S2 LINK: PASS`
