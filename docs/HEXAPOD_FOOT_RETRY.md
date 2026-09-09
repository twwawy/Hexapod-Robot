# 걸린 swing 발의 후퇴·재진입

2026-09-09. Adaptive MJX 전용 변경. STM32/manual 및 legacy 18-D는 변경하지 않았다.

## 데이터에서 확인한 내용

`hybrid-free-wave-recovery/02_hybrid-stair5_try00`의 영상과 local W&B 로그를 확인했다.
102400/204800 영상은 no_progress 종료, 307200 영상은 20초 영상 끝까지 실행 중이었다.
307200의 영상 종료를 실제 에피소드 완주로 해석하면 안 된다.

최신 완료 checkpoint **409600**, 기록된 **bd979c9** 소스, 영상과 같은 seed **20240**으로
별도 GPU rollout을 실행했다. 7.52~8.52초에 수직 계단 면 접촉이 기록되었다.
기존 접촉 함수는 normal 방향을 구분하지 않아 이 접촉도 지지/착지로 사용했다.
이번 재현은 RF만 계속 걸린 사례와 완전히 같지는 않았다. 최종적으로 RM/RB의
reach limiter가 작동하고 planner HOLD/no_progress로 끝났다.

원본 자료: `/home/huro/hexapod-diagnostics/rf-retract/rollout.json`, `replay.log`.
수정 후 비교 자료는 같은 디렉터리의 `modified/`에 별도로 저장한다.

## 변경 동작

- foot-to-world 접촉 normal을 world→foot 방향으로 정규화하여 수직 성분 >= 0.5일 때만
  load-bearing contact로 인정한다. 수직 벽 충돌은 착지와 구분한다.
  MuJoCo contact 방향을 사용하는 시뮬레이션 구현이며 실기 FSR과 동등하다는 주장은 아니다.
- active swing 중, 아직 착지하지 않은 발이 벽에 닿고 속도 < 1.5cm/s가 0.12초 지속되면
  retry 후보를 만든다. 발끝 wall contact 없이 정강이/구동부가 막힌 경우는
  command 발 위치와 실제 위치의 오차 > 2.5cm, 저속 상태 0.25초 지속으로 감지한다.
  정지한 발이라도 추종 오차가 없는 apex 대기는 재시도하지 않는다. phase 진행률 > 0.15, 나머지 다섯 발 접촉, fault 없음이 필요하다.
- 로봇 전방 반대 방향으로 **2.5 / 4 / 6cm 후퇴** 후보를 검사한다.
  초기 후퇴에 1.5cm lift를 결합한 뒤, 더 들어 올리고, 기존 착지점으로 전진·하강한다.
- 5개 waypoint의 piecewise quintic, 총 1.6초:
  `현재 실제 발 위치 → 뒤/위 → 후퇴 위치에서 상승 → 전방 상공 → 기존 endpoint`.
  RL action 및 기존 swing endpoint를 새 LiDAR값으로 덮어쓰지 않는다.
- 41개 경로 sample × 5개 footprint, 관측 여부, 지형 높이, 해당 발 IK/reach,
  나머지 다섯 발 IK/reach 및 CoM support margin >= 2cm를 검사한다.
  시작점의 벽 접촉은 보수적인 footprint height envelope 안에 있을 수 있으므로
  후퇴 구간에서는 초기 겹침을 악화시키지 않는 탈출만 허용한다. 후퇴 이후에는
  일반 경로 clearance 검사로 돌아간다. unknown 경로는 허용하지 않는다.
- 실행과 preflight는 **같은 trajectory 함수**를 사용한다. 실행 중 몸통 보행 이동과
  다른 다섯 발을 정지하고 posture를 고정한다. 매 tick 전발 IK/reach를 재검사한다.
- 다른 지지 발 접촉을 잃으면 중단하고, 명령이 유지되는 경우 기존 fault를 latch한다.
  발별 phase당 한 번만 시도한다. 검사 실패도 한 번의 시도로 기록한다.
  성공적으로 궤적을 마치면 원래 scheduler의 착지 확인/Late Landing으로 돌아간다.

이는 **지지 다섯 발이 확보된 swing 발의 충돌 복구**다. 여러 다리가 동시에
지지를 잃거나 이미 stance IK가 범위 밖인 상태까지 해결한다고 주장하지 않는다.
whole-body collision, 실제 충격/힘 제어는 포함하지 않는다.

## 관측과 checkpoint

24-D v4 및 observation path-v6 크기는 유지한다. 실제 동작이 달라졌으므로
소스 호환 검사는 유지하며 기존 bd979c9 checkpoint는 `--migrate-path-v6`로만
명시적 warm start한다. 전체 source manifest를 대조하고 optimizer는 새로 시작한다.
기존 53bab78/585bee2 및 검토된 v5 migration도 유지한다.

W&B metric: `foot_retry/active_s`, `foot_retry/rejected`.
영상의 `.termination.json`에는 실제 발 위치와 retry 상태/waypoint/시도 epoch를 추가했다.
viewer는 활성 retry 발과 상태를 표시한다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
python -m unittest mjx.tests.test_adaptive_foot_retry
```

PPO 학습이나 STM32 실기 검증은 수행하지 않았다. 수정 전후 rollout 결과는
진단 보고서에 구분해 기록한다. 단일 seed 결과를 학습 성능으로 일반화하지 않는다.

## 이번 검증 결과

- CPU unit/regression: `mjx.tests.test_adaptive_foot_retry` +
  `mjx.tests.test_adaptive_hybrid`, 21 tests 통과.
- 원본 409600 / seed 20240: 12.02초, x=0.4704m, Tripod/HOLD,
  RM/RB reach limit, no_progress 종료.
- 접촉 방향 분리 + 벽 접촉 retry 구현 후 같은 조건: 13.46초, x=0.5439m,
  Wave 실행 중, fault 없이 no_progress 종료.
- 두 rollout 모두 후퇴 재시도 자체는 발동하지 않았다. 따라서 실제 RF 탈출에
  성공했다고 볼 수 없으며, 전진 거리 차이를 재시도의 효과라고 주장하지 않는다.
- 이후 추가한 벽 접촉 없는 추종 오차 감지 조건은 unit test로만 확인했다.
  이 마지막 추가 후 전체 rollout은 다시 실행하지 않았다.

## 새 run으로 이어서 학습할 때

현재 실행 중인 pinned-source 학습에는 소스 수정이 자동 반영되지 않는다.
아래는 사용자가 실행하는 명령이며, 이번 작업에서 PPO는 실행하지 않았다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/train_adaptive_curriculum.sh \
  --profile hybrid --start-index 2 --perception teacher --command-mode terrain \
  --run-name hybrid-foot-retry-stair5 \
  --restore /home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum/hybrid-free-wave-recovery/02_hybrid-stair5_try00/checkpoints/000000409600 \
  --migrate-path-v6 \
  --timesteps-per-stage 800000 --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 9 --num-eval-envs 16 --episode-length 2000 \
  --action-profile terrain_mid --stair-clearance-extra 0.02 \
  --best-video-duration 20 --baseline-comparison-seconds 0 --discounting 0.997 \
  --promote-threshold 0.70 --max-retries -1 \
  --wandb --wandb-project hexapod-forward-completion --wandb-mode online
```
