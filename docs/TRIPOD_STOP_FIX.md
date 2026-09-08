# 수 회 Tripod 후 정지 수정

짧은 반복 학습과 cycle별 W&B 업로드는 [observe 프로필 명령](ADAPTIVE_OBSERVE_CYCLES.md)을 사용한다.
아래 `fast` 예시는 지형 승급 커리큘럼이며 같은 평지 반복용이 아니다.

## 확인한 원인과 수정

최근 로컬 `adaptive-flat-debug` / `adaptive-flat-fast` 학습 로그에는
`termination/no_progress`가 주된 종료 사유로 기록되어 있었다.
사용자가 본 특정 영상/가중치의 모든 실패를 동일 원인이라고 단정하지 않는다.

`AdaptiveGaitEnv._candidates()`는 classical half-stance 끝점에 예상 몸체
전진·회전까지 추가했다. 반면 실행기는 현재 몸체 좌표로 변환한 끝점을
phase 동안 고정한다. 전진 0.08 m/s, 1초 phase이면 기준점 전진량이
0.04 m여야 하는데 0.12 m가 되는 불일치였다. 이 추가 몸체 변환을 제거해
STM32 `FootTrajectory_BuildNominalPlan`의 half-stance 기준과 맞췄다.
terrain Z 조회, 후보 안전 검사, HOLD, no-progress 종료 조건은 유지한다.

추가 변경:

- Viewer가 checkpoint의 `action_profile`을 복원한다. `flat_safe`로 학습한
  출력을 viewer에서 갑자기 full authority로 실행하지 않는다.
- 공통 checkpoint metadata에도 `action_profile`을 기록한다.
- `policy_video.py`의 잘못된 들여쓰기와 reset 전 미정의 state/action 사용을 수정했다.
- 학습용·공통 GIF 생성기는 같은 이름의 `.termination.json`에 실제 종료 사유,
  성공 여부, 실행 시간과 요청 시간을 남긴다.
- `check_adaptive_rollout.py`는 사용자용 zero-action 진단 도구다. PPO를 실행하지
  않으며, 종료 사유와 완료한 phase 수를 JSON에 저장한다.

## 이미 실행한 확인과 한계

사용자의 검증 위임 지시를 어기고 수정 중 CPU 시뮬레이션을 실행했다.
사용자가 다시 지적한 뒤 시뮬레이션·학습·추가 테스트 실행을 중단했다.

실행했던 조건: flat, oracle, Tripod, action=0, seed=0, 명령 0.08 m/s.

| 항목 | 수정 전 | nominal 수정 후 |
|---|---|---|
| 완료 phase | 3 | 12 (Tripod 6 cycles) |
| 전진 위치 | 약 0.204 m | 약 1.000 m |
| 종료 시간 | 약 7.12초 | 약 14.34초 |
| 종료 | no_progress | 1 m 지형 목표 도달, failure 항목 없음 |

이는 특정 조건의 zero-action baseline 확인이다. 최근 PPO 가중치 재평가,
추가 학습, LiDAR 계단 등판, 실기는 수행하지 않았다. 이후 추가한 진단 도구와
영상 종료 JSON 생성 경로도 실행하지 않았다.

착지점은 phase-entry 몸체 좌표의 끝점이다. world marker와 실제 접촉점의 오차는
몸체 이동·자세 응답의 영향을 받는다. 이 수정으로 험지의 world-locked 착지나
전체 swept-path collision 검증까지 완료된 것은 아니다.

## 사용자가 실행할 명령

먼저 가중치 없이 기본 보행을 확인한다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/view_foothold_planner.sh --controller adaptive --terrain flat --perception oracle --gait-mode tripod --speed 0.08 --seed 0
```

실시간 viewer가 필요하지 않으면 curriculum 실행이 cycle 종료 후 best-score GIF를 자동 저장한다.

창 없이 동일 baseline의 종료 원인을 기록하려면:

```bash
python mjx/check_adaptive_rollout.py --cpu --perception oracle --terrain-level 0 --speed 0.08 --seed 0 --seconds 20
```

결과는 `mjx/generated/adaptive_baseline.json`에 저장된다. `completed_phases`는
한 Tripod group의 swing 수이므로 2 phase가 한 cycle이다. 1 m 목표 도달 후
`done=true`, `success=true`, 빈 `termination`이면 성공 종료다.
`termination/no_progress`이면 HOLD/전진 정체, tilt/clearance 등은 물리 실패다.

기본 보행을 확인한 뒤 새 출력 디렉터리로 학습한다. 아래 학습은 사용자가 실행한다.

```bash
bash scripts/train_adaptive_gait.sh --stage 1 --terrain-level 0 --perception teacher --action-profile flat_safe --output mjx/runs/adaptive-flat-nominal-fix
```

짧게 반복해 학습 변화만 보려면 아래 한 구간을 사용한다. 16초 episode는 flat에서
여러 Tripod cycle과 전진 정체를 볼 수 있는 최소 길이이고, 4회 평가는 100k step마다
수치를 남긴다. 각 실행은 새 output 경로를 사용한다.

```bash
bash scripts/train_adaptive_gait.sh \
  --stage 1 \
  --terrain-level 0 \
  --perception teacher \
  --action-profile flat_safe \
  --timesteps 400000 \
  --num-envs 128 \
  --batch-size 128 \
  --num-minibatches 4 \
  --episode-length 800 \
  --num-evals 4 \
  --best-video-duration 12 \
  --output mjx/runs/adaptive-flat-observe-01
```

다음 짧은 구간은 `--restore`에 직전 checkpoint 경로를 넣고 output만 바꾼다.

```bash
bash scripts/train_adaptive_gait.sh \
  --stage 1 --terrain-level 0 --perception teacher --action-profile flat_safe \
  --timesteps 400000 --num-envs 128 --batch-size 128 --num-minibatches 4 \
  --episode-length 800 --num-evals 4 --best-video-duration 12 \
  --restore mjx/runs/adaptive-flat-observe-01/checkpoints/<step> \
  --output mjx/runs/adaptive-flat-observe-02
```

여러 짧은 cycle을 자동으로 진행하면서 각 cycle의 최고 score 영상과 전체 최고 score 영상을
남기려면 curriculum runner를 사용한다. 아래는 cycle당 400k step, 16초 episode, 4회 평가다.

```bash
bash scripts/train_adaptive_curriculum.sh \
  --profile fast \
  --perception teacher \
  --run-name adaptive-observe-cycles \
  --timesteps-per-stage 400000 \
  --num-envs 128 \
  --num-evals 4 \
  --episode-length 800 \
  --action-profile flat_safe \
  --best-video-duration 12 \
  --wandb
```

저장 위치는 다음과 같다.

```text
mjx/runs/adaptive-curriculum/adaptive-observe-cycles/
  00_tripod-flat_try00/videos/best.gif       # 해당 cycle 내부 원본
  00_tripod-flat_try00/monitor/best_checkpoint.json # score/checkpoint manifest
  curriculum_history.json                     # cycle별 score/checkpoint/video 경로
```

W&B는 기본 활성화다. 매 평가 결과는 `eval/*`로 즉시 올라가고, 해당 cycle의 최고
score가 갱신될 때마다 `cycle/best_score`, `cycle/best_step`이 즉시 기록된다. cycle이
끝나면 같은 W&B run에 `cycle/best_video`와 checkpoint artifact가 업로드된다.

이전 checkpoint는 source hash가 달라 기존 strict loader에서 거부된다.
자동 호환 처리나 hash 우회는 추가하지 않았다. 이전 학습 결과는 보존하고,
수정된 실행기를 기준으로 새 학습 결과와 비교한다.

## 작업 범위

로컬에 있던 커리큘럼 및 학습 코드 변경은 보존했다. 이번 변경은 로컬 수정이며
추가 commit/push는 수행하지 않았다. STM32 및 기존 18-D 제어기는 수정하지 않았다.
