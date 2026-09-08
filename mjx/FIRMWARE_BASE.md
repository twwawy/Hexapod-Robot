# STM32 제어기 기반 MuJoCo GT base

2026-09-06 경로 구분: 이 문서는 현재 루트 MJX v4 기반이다. stage31 viewer는 가중치에 대응하는
격리된 v3 펌웨어 step을 사용하며, nominal gait를 유지하고 residual 입력 크기만 조절한다.
[실행 및 비교 방법](../docs/HEXAPOD_FOOTHOLD_PREVIEW_USAGE.md)을 참고한다.

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

RL scene은 원본 링크 질량과 관성을 비례 스케일해 로봇 전체를 `10.0 kg`으로 맞춘다.
18개 관절은 DS51150-270의 12.6 V 사양(357:1, stall `14.709975 Nm`, 무부하
`315.8 deg/s`)을 기준으로 한다. MuJoCo에는 `kp=500`, `kv=10`, output armature
`0.02 kg·m²`, damping `0.15 Nms/rad`, friction loss `0.8 Nm`을 적용한다. 뒤의
동역학 계수는 제조사 보증값이 아니라 실기 식별 전 calibration prior다.

`--terrain stairs`는 최종 curriculum과 같은 10단을 사용한다. 한 riser는 20 cm,
최상단은 바닥 기준 2 m다. 낮은 계단·울퉁불퉁 바닥·경사면은 RL environment가
같은 고정-shape scene에서 level별 collision geom을 선택한다.

## 파일

- `native/firmware_controller_bridge.c`: STM32 모듈 호출 순서와 단순 C API
- `firmware_controller.py`: C 소스 변경 감지, 공유 라이브러리 빌드, ctypes 연결
- `run_firmware_base.py`: MuJoCo GT 어댑터, actuator 좌우 부호 변환, 뷰어
- `firmware_mjx_controller.py`: 위 C 제어기를 GPU 병렬 학습용 JAX 상태로 옮긴 구현
- `terrain_curriculum.py`: level 0~16의 지형 치수·목표 위치를 한 곳에서 정의
- `servo_model.py`: DS51150 사양과 교체 가능한 calibration prior를 한 곳에서 정의
- `rough_terrain_env.py`: 펌웨어 base 위 18-D 발끝 residual과 안전 종료/계단 보상

기본 Swing 높이는 평지 기준 6 cm다. RL Z action은 Swing에서 4~25 cm 높이를
선택하지만 이륙·착지점 Z는 바꾸지 않는다. Stance에서는 ±100 mm까지 요청할 수 있으며
Late Landing은 정책 개입 없이 펌웨어가 처리한다.

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
