# 짧은 학습 cycle과 W&B

`--profile observe`는 같은 평지 Tripod를 지정한 횟수만큼 반복한다.
기존 `fast`는 16개 지형/gait stage의 승급 커리큘럼이므로 다른 용도다.
observe에서는 승급 성공률 때문에 3회 시도 뒤 중단되지 않는다.
실제 학습 오류·영상 저장 실패는 로그를 남기고 중단한다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/train_adaptive_curriculum.sh \
  --profile observe --cycles 5 \
  --perception teacher --run-name adaptive-observe-cycles \
  --timesteps-per-stage 400000 \
  --num-envs 128 --batch-size 64 --num-minibatches 4 \
  --num-evals 5 --num-eval-envs 4 \
  --episode-length 800 --action-profile flat_safe \
  --best-video-duration 12 --wandb --wandb-mode online
```

- 한 cycle: 약 400k 환경 step. PPO batch 배수로 올림될 수 있다.
- 평가: 초기 1회 + 학습 중 4회. `num-evals 4`는 학습 중 3회였음.
- 800 policy step은 최대 16초 episode. 조기 실패 또는 목표 도달 시 먼저 종료한다.
- cycle별 최고 score는 `eval/episode_reward` 기준이다. 초기 미학습 평가는 best에서 제외한다.
- 매 평가 callback에서 `cycle/best_score`, `cycle/best_step`과 summary를 W&B에 기록한다.
  실시간은 평가 결과가 나올 때의 업로드를 뜻하며 매 물리 tick의 평가를 뜻하지 않는다.
- cycle 종료 직후 `cycle/best_video`를 같은 run에 업로드한다. 최대 12초 GIF이며
  성공·실패로 일찍 끝나면 `.termination.json`에 이유를 남긴다.
- artifact에는 GIF, 실제 best checkpoint 디렉터리, metrics, best pointer,
  영상 종료 보고서를 포함한다. 별도의 전체 최고 선정은 없다.
- 모든 cycle은 같은 W&B group 아래 별도 run이고 seed도 고정해 비교한다.
- 다음 cycle은 직전 cycle의 best 가중치와 관측 정규화를 복원한다.
  PPO optimizer 상태까지 연속 유지하는 방식은 아니며 subprocess마다 재초기화된다.
- 같은 run-name으로 다시 실행하면 기존 결과를 보존하고 timestamp를 붙인 새 실험을 만든다.

저장 루트: `mjx/runs/adaptive-curriculum/<실험명>/`.
각 `00_tripod-flat_try00/` 등의 디렉터리에 `videos/best.gif`, checkpoints, monitor 파일이 있다.
상위 `cycle-00-try00.log`에는 학습·렌더링 오류를 포함한 표준 출력을 저장하며,
`active_cycle.json`에는 마지막 실행 명령을 기록한다.

400k step은 비교를 위한 시작 설정이며 유의미한 개선을 보장하는 최소 학습량은 아니다.
W&B 인증은 현재 머신의 로그인 계정을 사용한다. 이전 소스 checkpoint의 strict hash
검사를 우회하지 않는다. 이번 변경의 학습·시뮬레이션·렌더링·업로드는 실행하지 않았다.

## 명령어 조절표

아래 표는 **curriculum wrapper** `scripts/train_adaptive_curriculum.sh`에서 지원하는 옵션이다.
명령 마지막의 `\`는 다음 줄도 같은 명령이라는 뜻이며, 마지막 옵션 뒤에는 붙이지 않는다.
변경하려는 옵션의 숫자나 값을 바꾸면 된다.

| 조절 목적 | 옵션 | 기본값 | 사용 예시 / 영향 |
|---|---|---|---|
| 같은 평지에서 반복 | `--profile` | `fast` | 반복 관찰은 반드시 `observe`; `fast`/`full`은 지형 승급 |
| 반복 횟수 | `--cycles` | `5` | `observe` 전용. `3`이면 세 cycle, `10`이면 열 cycle |
| cycle 하나의 학습량 | `--timesteps-per-stage` | `10000000` | `400000`에서 시작. `800000`이면 cycle당 학습량 두 배 |
| 동시에 학습하는 환경 수 | `--num-envs` | `64` | 예시 `128`. 늘리면 메모리도 증가하며 속도 개선은 하드웨어에 따라 다름 |
| PPO minibatch 크기 | `--batch-size` | `64` | 환경 수와 다른 의미. 기본 유지 권장 |
| PPO minibatch 개수 | `--num-minibatches` | `4` | `batch-size × num-minibatches`가 `num-envs`로 나누어떨어져야 함 |
| 평가·checkpoint 빈도 | `--num-evals` | `10` | `5` = 초기 1회 + 학습 중 4회. 늘리면 score 갱신이 잦아지지만 평가 비용 증가 |
| 한 번에 평가하는 환경 수 | `--num-eval-envs` | `4` | 실제로는 `min(num-envs, num-eval-envs)`. `8`이면 평가 표본을 늘림 |
| episode 최대 길이 | `--episode-length` | `2000` | 정책 step 단위. `800` = 16초, `1000` = 20초. 실제 대기 시간이 아님 |
| RL 조절 권한 | `--action-profile` | `full` | 평지 초기에는 `flat_safe`; 아래 표 참고 |
| 영상 최대 길이 | `--best-video-duration` | `12` | 초 단위. `20`으로 늘려도 학습량은 같고 렌더링 비용만 증가 |
| 센서 입력 종류 | `--perception` | `teacher` | `teacher` = GT 지형, `lidar` = LiDAR map. teacher 결과를 LiDAR 성공으로 해석하지 않음 |
| 비교 seed | `--seed` | `40` | observe는 cycle마다 같은 seed; 실험 간 seed를 바꾸어 비교 가능 |
| 실험 이름 / W&B group | `--run-name` | `adaptive-v4-curriculum` | 예: `flat-400k-5cycles`. 같은 이름 재사용 시 timestamp 자동 추가 |
| 로컬 결과 상위 경로 | `--run-root` | `mjx/runs/adaptive-curriculum` | 예: `--run-root /home/huro/hexapod-runs` |
| 기존 가중치에서 시작 | `--restore` | 없음 | 호환되는 checkpoint 또는 run 경로. 기존 결과를 덮어쓰는 resume은 아님 |
| teacher로 LiDAR 초기화 | `--init-teacher` | 없음 | `--perception lidar`에서만 사용. `--restore`와 동시 사용 불가 |
| W&B 연결 | `--wandb` | 켜짐 | 명령에 명시해 두면 연결 의도가 분명함. `--no-wandb`는 명시적 비활성 |
| W&B project | `--wandb-project` | `hexapod-adaptive-gait` | 원하는 project 이름 |
| W&B 계정/팀 | `--wandb-entity` | 현재 로그인 기본값 | 원하는 사용자명 또는 팀 이름 |
| W&B 전송 모드 | `--wandb-mode` | `online` | 실시간 업로드는 `online`. `offline`/`disabled`는 실시간 업로드하지 않음 |
| 시작 stage 인덱스 | `--start-index` | `0` | observe도 해당 인덱스부터 마지막까지 실행. 이전 가중치는 자동 탐색하지 않으므로 `--restore` 필요 |
| 지형 승급 지표 | `--promote-key` | `eval/episode_terrain_success` | `fast`/`full` 전용; cycle best 선정 지표와 다름 |
| 승급 기준 | `--promote-threshold` | `0.70` | `fast`/`full` 전용, 범위 0~1 |
| 승급 실패 추가 시도 | `--max-retries` | `2` | `fast`/`full`: 최초+추가 2회. observe는 승급 검사 없이 다음 cycle |

`--cycles`는 보행의 Tripod cycle 수가 아니라 **학습 실행 구간 수**다.
총 요청 학습량은 observe에서 `cycles × timesteps-per-stage`이고, 병렬 환경 수를 다시 곱하지 않는다.
예시 5 × 400000 = 200만 환경 step이며, 평가·영상 생성 step은 별도다.
`num-evals 5`의 평가 간격은 약 `timesteps-per-stage / 4`이며 batch 단위 올림 때문에 정확히 100000마다일 필요는 없다.
`num-evals 1`도 구현상 학습 후 평가 1회를 수행한다.

## RL 권한 조절

| profile | landing XY / clearance action 배율 | body / stride / timing action 배율 |
|---|---|---|
| `flat_safe` | 0.10 | 0 (classical baseline) |
| `terrain_mid` | 0.50 | 0.35 |
| `terrain_high` | 0.75 | 0.65 |
| `full` | 1.00 | 1.00 |

배율은 policy 출력에 적용되며 geometry/safety projection을 통과한 값만 실행한다.
작은 XY 요청은 local candidate 격자에 투영되면서 reference와 동일한 착지점이 될 수 있다.
`flat_safe`를 전체 험지 커리큘럼에 고정하면 자세·보폭·timing을 학습할 수 없다.
observe에서 평지 기본 보행을 먼저 비교하고, 지형 적응 학습으로 넘어갈 때 profile을 선택한다.

## 자주 바꾸는 조합

기본 명령에서 다음 옵션만 교체한다. 같은 옵션을 여러 번 적기보다 기존 값을 바꾼다.

| 원하는 실험 | 바꿀 옵션 |
|---|---|
| 짧게 세 구간만 비교 | `--cycles 3 --timesteps-per-stage 400000` |
| 한 구간을 더 충분히 학습 | `--cycles 5 --timesteps-per-stage 800000` |
| score를 더 자주 확인 | `--num-evals 9` (초기+학습 중 8회) |
| 평가 표본을 더 확보 | `--num-eval-envs 8` |
| GPU 메모리 부담 완화 시도 | `--num-envs 64 --batch-size 64 --num-minibatches 4` |
| 긴 보행과 영상 비교 | `--episode-length 1500 --best-video-duration 25` |
| 다른 실험과 분리 | `--run-name flat-800k-seed41 --seed 41` |

episode/영상 길이를 늘려도 성공·실패의 조기 종료 조건은 유지된다.
짧은 실행이라도 최초 JAX 컴파일과 cycle 종료 렌더링 시간이 필요하다. 일정한 분 단위 완료 시간을 보장하지 않는다.

## 가중치 이어받기

각 cycle의 `monitor/best_checkpoint.json`에 best checkpoint 경로가 기록된다.
그 경로를 `--restore`에 지정하고 새 실험명으로 시작한다. run 디렉터리를 지정하면
그 안의 **가장 최근** checkpoint가 선택되므로, best를 원할 때는 JSON의 정확한 경로를 쓴다.

```bash
# 기존 기본 명령에 다음 두 옵션을 지정한다.
--run-name adaptive-observe-next
--restore /실제/경로/checkpoints/000000000000
```

위 경로는 예시 자리표시자다. metadata에 기록된 실제 경로로 교체한다.
같은 소스 계약만 복원 가능하며 소스가 바뀌었다는 오류를 무시하거나 hash를 수정하지 않는다.
가중치와 관측 정규화만 복원하고 optimizer는 새로 시작한다.

## W&B에서 보는 항목과 영상 저장 시점

| 시점 | 항목 |
|---|---|
| 매 평가 완료 | `eval/*`, `training/*`, 현재 cycle의 `cycle/best_score`, `cycle/best_step` |
| cycle 완료 | 해당 cycle best의 `cycle/best_video` 및 policy artifact |
| 오류 발생 | 로컬 `cycle-NN-tryNN.log`, 렌더링 실패 시 cycle 내부 `monitor/best_video_error.json` |

score는 cycle별 summary에도 기록되므로 run 목록에서 비교할 수 있다.
영상은 평가마다 생성하지 않고 cycle 종료 직후 한 번 생성·업로드한다.
W&B의 네트워크 전송 시간은 별도로 걸릴 수 있다. 전체 best를 선별하거나 합치는 코드는 없다.

## 추가 학습 옵션이 필요할 때

한 구간용 `scripts/train_adaptive_gait.sh`에는 `--learning-rate`, `--entropy-cost`,
`--score-key`, `--video-fps`, `--video-width`, `--video-height`, LiDAR sampling 옵션이 있다.
이 옵션은 현재 curriculum wrapper가 전달하지 않으므로 wrapper 명령에 바로 붙이면 안 된다.
전체 옵션 설명만 출력하는 명령은 다음과 같다. 학습을 시작하지 않는다.

```bash
bash scripts/train_adaptive_curriculum.sh --help
bash scripts/train_adaptive_gait.sh --help
```

공식 반복 학습 진입점은 `train_adaptive_curriculum.sh`다.
이름이 비슷한 `mjx/train_adoptive_curriculum.py`는 기존 별도 파일이며 위 명령에서 사용하지 않는다.
