import mujoco
import mujoco.viewer
import time

# 1. 아주 단순한 로봇(빨간 상자) 설계도 문자열 (MJCF)
xml_string = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/> <body pos="0 0 2"> <joint type="free"/> <geom type="box" size="0.2 0.2 0.2" rgba="1 0 0 1" mass="1"/> </body>
  </worldbody>
</mujoco>
"""

# 2. 엔진에 설계도 로드
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# 3. 뷰어 실행 및 시뮬레이션 루프 시작
with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"MuJoCo 버전: {mujoco.__version__} 정상 작동 중!")
    
    while viewer.is_running():
        # 물리 스텝 1회 진행
        mujoco.mj_step(model, data)
        # 뷰어에 렌더링 업데이트
        viewer.sync()
        # 연산 속도가 너무 빨라 눈에 보이게 하기 위한 딜레이
        time.sleep(0.01)
