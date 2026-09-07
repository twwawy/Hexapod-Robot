# 조종기 명령을 전제로 하는 학습

## 구현 범위

상위 입력은 `[vx, wz]`이고 다리/관절 실행은 위치 기반이다. RC 모드는
전진·후진·제자리 좌/우 회전·전후진 곡선·정지 명령을 입력받는 residual policy를 학습한다.
학습 중 물리 조종기를 연결하는 대신 JAX fixed-shape sampler로 조작 명령을 생성한다.
실제 RC 수신기/Jetson/STM32 통신 연결을 추가한 작업은 아니다.

`--command-mode terrain`은 기존 전진 등판 task, `--command-mode rc`는 자유 조종 추종 task다.
작업 구분은 checkpoint metadata에 저장하며 다른 task의 가중치를 자동 restore하지 않는다.
24-D action과 CNN 관측 크기, terrain_mid residual 권한은 유지한다. 과거 source-hash
계약의 checkpoint는 이전 코드에서 재생해야 한다. 이미 실행 중인 PPO를 자동 변경하지 않는다.
이 변경의 학습·rollout·unit test는 실행하지 않았다.

## 조종 명령과 목표

각 목표 명령은 2~5초 유지한다. vx는 ±0.03~0.08m/s, yaw는 ±0.12~0.25rad/s,
정지/전진/후진/제자리 좌회전/우회전/네 방향 전후진 곡선을 균등한 9개 mode로 뽑는다.
slew 한계는 vx 0.08m/s², yaw 0.4rad/s²이며 deadband는 0.005m/s, 0.02rad/s다.
deadband 이전의 별도 slew 상태를 저장하여 작은 tick마다 0으로 되돌아가는 오류를 피한다.
지형/물리 외란 RNG와 명령 RNG를 분리하여 paired baseline/policy의 명령 sequence가 같도록 한다.

현재 action을 계산한 뒤 command를 몰래 바꾸지 않는다. step 종료에 **다음 observation의
command를 갱신**해 다음 policy tick에서 그대로 사용한다. 초기 command delay 동안 0이다.
Viewer는 checkpoint의 task를 따르되 `external_commands=True`로 자동 sampler를 끈다.
그때는 기존 방향키/WASD가 명령을 공급한다. 실물 조종기 입력은 같은 command 경로로
연결해야 하며 실제 전처리 deadband/slew와 학습 설정의 일치는 후속 검증 대상이다.

## 보상과 실패 기준

- velocity reward는 요청한 **부호 있는 vx**와 실제 body-forward 속도를 비교한다.
- progress/under-speed는 `[vx, 0.3*wz]` 공간의 명령 방향 투영으로 계산한다.
  yaw를 선형 속도로 비교하는 0.3m는 reward/watchdog용 characteristic radius이며 실측 몸체 치수가 아니다.
- 후진·제자리 회전이 world X 증가 없음 때문에 no-progress로 종료되지 않는다.
- 완전 정지 명령에는 no-progress watchdog을 비활성화한다. 정지 상태에서 움직이지 않는 것은 정상이다.
- RC에서는 고정 world-X goal 및 상승 보너스를 사용하지 않는다. 안전하게 episode horizon을
  마치고, 명령 활성 시간의 70% 이상에서 선형 오차 ≤0.02m/s 및 yaw 오차 ≤0.07rad/s이면
  `command_success=1`이다. 전환 중 추종 오차도 포함한다. 정지로 저속 명령을 통과할 수
  없도록 최소 목표 속도보다 작은 오차 문턱을 사용한다.
- 안전 실패와 3초 no-progress는 계속 종료하며, 미달한 horizon 종료는 timeout이다.
  adaptive 완주/실패 reward 크기는 유지하되 RC의 +120은 command task 성공 의미다.
- `terrain_success`는 RC에서 0이다. RC 성공을 계단 등판 성공이라고 기록하지 않는다.

`command_success`의 70% 시간 기준과 curriculum 승급의 70% episode 성공률은 서로 다른 조건이다.
16 eval env 중 최소 12개(75%)가 command task에 성공해야 기본 승급 문턱 70%를 넘는다.
작은 표본 및 랜덤 명령이므로 실기 조종 성능을 보장하는 기준은 아니다.

## 먼저 평지 RC 학습

자유 회전/후진을 좁은 일방향 stair course에 그대로 넣으면 계단 밖으로 돌아가는 정상 명령과
등판 목표가 충돌한다. 따라서 `--profile rc`는 평지 Tripod → 평지 Wave → 평지 Hybrid다.
무한 retry는 명령 성공률을 기준으로 한다. 거친 지형에서의 조종+등판 혼합 task는 아직
구현하지 않았으며, 이후 command-conditioned terrain course/명령 분포를 함께 설계해야 한다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/train_adaptive_curriculum.sh \
  --profile rc --command-mode rc --perception teacher \
  --run-name adaptive-rc-gt-flat \
  --timesteps-per-stage 800000 \
  --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 5 --num-eval-envs 16 --episode-length 2000 \
  --action-profile terrain_mid --discounting .997 \
  --baseline-comparison-seconds 20 --best-video-duration 40 \
  --promote-key eval/episode_command_success --promote-threshold .70 \
  --max-retries -1 --on-stage-failure stop \
  --wandb --wandb-mode online
```

새 task이므로 위 명령은 restore 없이 시작한다. `--episode-length 2000`은 40초로
여러 명령 전환을 포함한다. 기존 계단 학습 run/checkpoint를 삭제하지 않는다.

## 기록

- cycle best는 `command_success` 우선, 같은 성공률이면 reward로 선택한다.
- W&B cycle별 best score/영상/40초 최대 길이 및 정지 직전 진단은 유지한다.
- 영상 상단에 command task, vx, yaw를 표시한다. 정지 명령 구간과 제어기 정지를 구분할 수 있다.
- `eval/episode_command_success`는 episode 성공 비율이다.
- `command/linear_error_mps`, `command/yaw_error_radps`, `command/vx`, `command/wz`는
  per-tick 지표다. Brax episode 값은 합산되므로 평균을 보려면 episode length로 나눈다.
- `command_good_fraction`은 매 tick의 누적 good fraction이다. episode 합을 그대로 성공률로
  해석하지 않는다. 비교 진단의 linear/yaw error는 실제 실행 tick수로 나눈 평균이다.
- RC 비교의 `forward_m`는 출발 방향 순변위로, 후진/회전을 포함한 policy의 우열 지표가 아니다.
  `comparison/*/command/linear_error_mps`, `comparison/*/command/yaw_error_radps`를 우선 본다.

사용자용 단위 검사: `python -m unittest mjx.tests.test_operator_commands`.
부호 있는 후진/yaw/정지 보상, 정지로 저속 명령 통과 방지, deadband에서의 slew 탈출,
fixed-size batched sampler를 검사한다. 이번 작업에서는 실행하지 않았다.
