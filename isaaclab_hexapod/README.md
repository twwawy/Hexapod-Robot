# Hexapod Isaac Lab port

2026-09-06 문서 동기화: 아래는 Isaac 개발 경로(v4)이며, stage31 v3 MuJoCo 뷰어와 별개다.
[최신 보행 실행](../docs/HEXAPOD_FOOTHOLD_PREVIEW_USAGE.md), [GT/센서 학습 설계](../docs/HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md)를 참고한다.
기존 미커밋 소스·USD·handoff를 함께 반영했으며 이번 통합에서 Isaac 실행/학습 검증은 다시 수행하지 않았다.

현재 이 디렉터리에는 최신 MJX 학습 handoff, CAD mesh를 보존한 단일-root USD,
batch Torch firmware controller, LiDAR/Depth/IMU elevation-map pipeline과 asymmetric
RSL-RL 설정이 들어 있습니다.

현재 residual-v4 action은 18-D를 유지하면서 swing X/Y와 stance Z에 축별 최대
`100 mm` 요청 권한을 줍니다. action은 누적 delta가 아니라 직전 controller state로
0.1초 smoothing되는 절대 residual입니다. swing Z는 기존 4~25 cm 높이 의미를 유지합니다.
기존 residual-v3 checkpoint는 이 scale과 호환되지 않아 자동 로드하지 않습니다.

현재 모델/지형을 Isaac Sim GUI에서 열려면 다음을 실행합니다.

```bash
cd /home/huro/Hexapod-Robot
./isaaclab_hexapod/scripts/build_asset.sh
./isaaclab_hexapod/scripts/run_realtime.sh
```

첫 명령은 새 URDF의 133개 CAD mesh를 visual로 유지하고, MJX 학습과 같은 25개 primitive
collider를 별도로 둔 USD를 생성합니다. 두 번째 명령은 최신 handoff에 기록된 terrain
level을 기본으로 사용합니다. 다른 level을 보려면 `--terrain-level 5`처럼 지정합니다.
실시간 경로는 full-CAD robot 1대와 새 URDF 장착 좌표의 MID-360 RTX proxy를 생성합니다.
LiDAR 없이 모델만 확인하려면 `--no-rtx-lidar`를 붙입니다.

RTX 3090 학습 부하 확인은 기본 512 environments로 시작합니다.

```bash
HEXAPOD_NUM_ENVS=512 ./isaaclab_hexapod/scripts/train_perceptive_gpu80.sh
watch -n 1 nvidia-smi
```

GPU 사용률이 70% 아래면 `768`, OOM 또는 95% 이상이면 `384`/`256`으로 조절합니다.
환경 수가 GPU 부하를 결정하므로 80%를 정확히 고정하는 옵션은 아닙니다. 또한 현재
Isaac task reward는 개발 scaffold이므로 이 명령은 부하/통합 확인용이며, 최신 MJX
reward와 termination 이식 전 생성된 policy를 실사용 학습 결과로 취급하면 안 됩니다.

최신 평가가 안전 gate를 통과하지 못했기 때문에 MJX checkpoint 자동 로드는 꺼져
있습니다. 현재 handoff와 정확한 사유는
[`data/training/latest_mjx_training.json`](data/training/latest_mjx_training.json)에
기록됩니다.

설치, 생성물, 실행법, 확인 결과와 남은 작업은
[PORT_RESULT_AND_USAGE.md](PORT_RESULT_AND_USAGE.md)에 정리되어 있습니다.
