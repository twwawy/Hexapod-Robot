# STM32 제어기 기반 MuJoCo GT base

## 목적

강화학습 residual을 붙이기 전에 pull한 STM32 제어기만으로 평지와 단차에서
동작하는 기준선을 만든다. 제어 알고리즘을 Python으로 다시 해석하지 않고
`SW/STM32/workspace/Hexapod/Core/Src/high_control`의 C 소스를 직접 빌드한다.

## 실행 파이프라인

```text
목표 vx, wz
    ↓
DroneController (짐벌 LPF·heading reference)
    ↓
GaitPoseController ← GT body position / yaw
    ↓
WorkspaceLimiter
    ↓
GaitManager ← GT foot contact
    ↓
FootTrajectory (Early/Late Landing)
    ↓
BodyPostureController ← GT roll / pitch
    ↓
LegKinematics + joint rate limit
    ↓
MuJoCo 18 position actuators
```

현재 STM32의 `BodyPositionEstimator`는 사용하지 않는다. MuJoCo freejoint의 위치를
직접 넣어 추정기 오차와 제어기 자체 동작을 분리한다. 관절각과 발 접촉 역시
MuJoCo 상태와 contact pair에서 직접 읽는다. 다음 단계에서 GT를 센서 모델로
교체할 때 이 어댑터 경계만 바꾸면 된다.

계단은 7단이며 최상단 높이는 바닥 기준 정확히 20 cm다. 따라서 한 riser는
`20/7 ≈ 2.86 cm`이고 펌웨어 baseline과 RL level 4가 같은 형상을 사용한다.

## 파일

- `native/firmware_controller_bridge.c`: STM32 모듈 호출 순서와 단순 C API
- `firmware_controller.py`: C 소스 변경 감지, 공유 라이브러리 빌드, ctypes 연결
- `run_firmware_base.py`: MuJoCo GT 어댑터, actuator 좌우 부호 변환, 뷰어
- `firmware_mjx_controller.py`: 위 C 제어기를 GPU 병렬 학습용 JAX 상태로 옮긴 구현
- `rough_terrain_env.py`: 펌웨어 base 위 18-D 발끝 residual과 안전 종료/계단 보상

생성되는 `generated/libhexapod_firmware_controller.so`는 빌드 산출물이므로 Git에
포함하지 않는다. STM32 high-control 소스가 변경되면 다음 실행 때 자동으로 다시
빌드된다.

## 실행

```bash
.venv/bin/python mjx/run_firmware_base.py
.venv/bin/python mjx/run_firmware_base.py --terrain stairs
MUJOCO_GL=egl .venv/bin/python mjx/run_firmware_base.py --headless
```

기본 명령은 1초 정지 후 `0.08 m/s` 전진이다. `--speed`, `--yaw-rate`,
`--command-delay`, `--duration`으로 변경할 수 있다.

기본 실행은 RL과 같은 controller-failure 조건(IK invalid, 관절 한계 1° 이내,
과속, 45° 기울기, 낮은 clearance, 비정상 수치)을 만나면 폭발 전에 멈추고 이유를
출력한다. 실패 이후까지 관찰할 때만 `--allow-unsafe`를 사용한다.
