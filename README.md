# Hexapod Robot — 전진 지형 완주 학습

6개 다리·18개 관절을 사용하는 로봇의 MJX 기반 residual RL 프로젝트다.
현재 목표는 **yaw 조종 명령 없이 전진해서 지형을 완주하는 정책**을 학습하는 것이다.
작업 브랜치는 `codex/adaptive-hybrid-rl-integration`이다.

## 계단 경로 보정 업데이트

계단 경로의 최소 여유 높이와 그 위치·시점을 계산해 높이/이동 timing 보정안을 비교한다.
기존 보폭 후보에도 같은 검사를 적용한다. [설계·진단·사용자 검사](docs/ADAPTIVE_PATH_BOTTLENECK.md).
관측 계약은 path v6다. 지정 계단 v5 checkpoint는 아래 명시적 이전 스크립트로 이어갈 수 있다.
실행 중인 pinned source 학습에는 변경이 적용되지 않는다.

## 현재 학습 방향

| 항목 | 설정 |
|---|---|
| Task | `terrain`: 전진 지형 완주 |
| Curriculum | `teacher`: GT 지형 기반 |
| Yaw 명령 | 0; 임의 좌우 회전·후진·정지 명령 sampler 미사용 |
| Yaw 안정화 | 직진 중 불필요한 회전을 억제하는 기존 보상 유지 |
| Residual | `terrain_mid` 유지 |
| 승급 | `eval/episode_terrain_success ≥ 0.70` |
| 실패 시 | 같은 단계에서 무제한 retry |
| 초기 가중치 | RC checkpoint를 restore하지 않고 새로 시작 |

Tripod 순서는 **flat → ramp8 → stair5 → stair8 → rough25 → ramp15 → stair10 → rough50**다.
이후 teacher profile에 정의된 Wave/Hybrid 단계로 진행한다.
현재는 GT로 보행을 학습하며 LiDAR 입력 학습과 실기 연결은 후속 단계다.

## 제어 구조

```text
전진 명령 + 로봇 상태 + GT 지형
                ↓
Geometry planner / elevation grid
                ↓
Classical gait + 24-D RL residual
                ↓
Safety projection / gait supervisor
                ↓
발끝 목표 → IK → 관절 위치 목표
```

상위 입력은 속도 명령이고 다리·관절 실행은 위치 기반이다.
Geometry는 착지 Z·안전 후보·필요 clearance를 계산한다.
RL은 착지 XY·clearance·Roll/Pitch·몸체 높이·보폭·apex/transfer timing을 조절한다.
24-D action 계약과 residual 범위는 이번 변경에서 바꾸지 않았다.
접촉·IK·workspace·gait 전환 안전은 제어기가 담당한다.

## 시작 명령

기존 RC 학습이 실행 중이면 먼저 종료한다. 첫 실행에는 `--restore`를 넣지 않는다.
동일 완주 task의 retry/승급에서는 manager가 checkpoint를 자동으로 이어받는다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate

bash scripts/train_adaptive_curriculum.sh \
  --profile teacher \
  --command-mode terrain \
  --perception teacher \
  --run-name forward-gt-path-v6 \
  --timesteps-per-stage 800000 \
  --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 5 --num-eval-envs 16 --episode-length 8000 \
  --action-profile terrain_mid --discounting 0.997 \
  --baseline-comparison-seconds 20 --best-video-duration 40 \
  --promote-key eval/episode_terrain_success --promote-threshold 0.70 \
  --max-retries -1 --on-stage-failure stop \
  --wandb --wandb-mode online
```

## 설정 조절

| 옵션 | 의미 |
|---|---|
| `--timesteps-per-stage` | 각 stage 시도당 학습량; 실제 값은 PPO batch 단위에 따라 달라질 수 있음 |
| `--num-envs` | 병렬 학습 환경 수 |
| `--batch-size`, `--num-minibatches` | PPO batch 구성 |
| `--num-evals`, `--num-eval-envs` | 평가 횟수·평가 환경 수 |
| `--episode-length 8000` | 최대 160초 episode; 완주/실패하면 더 일찍 종료 |
| `--best-video-duration 40` | 최대 40초 영상; 학습 episode 길이와 별개 |
| `--max-retries -1` | 완주율 기준을 통과할 때까지 재시도 |

## W&B와 결과 확인

Cycle은 `tryXX` 하나의 학습 시도다. 평가 중 점수·완주율을 기록하고,
매 학습 후 평가 종료마다 그 시점까지의 cycle best checkpoint로 영상을 생성해 W&B에 업로드한다.
Best가 갱신되지 않은 평가에서도 업로드한다. Step 0 평가와 cycle 종료의 중복 영상은 제외한다.
Best는 완주율 우선, 동률이면 reward로 선택한다. 전체 누적 best 영상은 생성하지 않는다.

- `eval/episode_terrain_success`: 승급 기준. 평가 환경 16개에서는 최소 12개 성공이 필요하다.
- `cycle/best_video_status`: 영상 생성·업로드 진행 또는 실패 상태.
- Cycle 영상 및 baseline 비교: 전진·접촉·정지 원인과 residual 효과 확인.
- 40초 영상이 끝났다는 이유만으로 160초 episode의 완주 여부를 판단하지 않는다.

결과는 `mjx/runs/adaptive-curriculum/<run-name>/` 아래에 저장한다.
Cycle 영상은 각 시도 폴더의 `videos/best.gif`다.

## 소스 고정과 checkpoint

Shell 진입점은 시작 commit을 `~/.cache/hexapod-training-sources/`의 독립 worktree에
고정한다. 개발 저장소를 수정해도 같은 run의 retry에는 섞이지 않는다.
시작 로그의 `PINNED SOURCE`, `SOURCE REVISION`을 보관하고 실행 중 해당 경로를 수정/삭제하지 않는다.
추적 파일의 미커밋 변경은 시작 전에 commit해야 한다. Untracked 의존 파일은 복사되지 않는다.
직접 Python trainer/manager 실행에는 이 고정 기능이 적용되지 않는다.

RC와 terrain은 서로 다른 task다. RC 가중치를 자동 이전하지 않고 checkpoint 계약 검사를 유지한다.

## 코드와 문서

| 경로 | 역할 |
|---|---|
| `scripts/train_adaptive_curriculum.sh` | 권장 학습 진입점 |
| `mjx/launch_adaptive_curriculum.py` | 소스 고정 |
| `mjx/train_adaptive_curriculum.py` | 지형 단계·retry·승급 |
| `mjx/train_adaptive_gait.py` | PPO·평가·W&B·영상 |
| `mjx/adaptive_gait_env.py` | 관측·보상·학습 환경 |
| `mjx/adaptive_gait_controller.py` | Residual·궤적·실행 목표 |
| `mjx/adaptive_foothold_estimator.py` | 착지 후보 계산 |
| `SW/STM32/`, `SW/Jetson/`, `HW/` | 실기 제어·상위 연결·기구 자료 |

상세: [GT 전진 완주 학습](docs/ADAPTIVE_GT_TEACHER.md).
학습·시뮬레이션·실기 검증은 사용자가 실행한다. 설정 변경을 학습 성능 검증으로 취급하지 않는다.

매 평가 영상 및 기존 계단 checkpoint 재개: [실행 안내](docs/ADAPTIVE_EVAL_VIDEOS.md).

## 기존 계단 best에서 v6로 이어가기

```bash
bash /home/huro/Hexapod-Robot-integration/scripts/resume_stair5_path_v6.sh
```

이 명령은 terrain 5부터 v6 경로 보정을 적용하며 `--migrate-path-v6`로 지정 v5 가중치를 이전한다.
기존 feature 가중치·정규화 통계·CNN을 보존하고 추가 feature 입력 가중치를 0으로 초기화한다.
Optimizer는 새로 시작한다. 매 평가 best-so-far 영상과 W&B 기록을 유지한다.
변환·학습은 사용자 실행 대상이며 이전 정책의 보행 성능 보존을 보장하지 않는다.
이 목적에는 별도 `Hexapod-Robot-stair5-resume`의 v5 재개 스크립트를 사용하지 않는다.
