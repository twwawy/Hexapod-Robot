# 완주 중심 보상과 비동기 사용자 실행

재접촉 수정 이후의 최신 추천은 [70% 통과까지 같은 지형 재시도](ADAPTIVE_RECONTACT.md)다.
아래 advance는 이전 실행 설정이며 필요할 때만 명시적으로 선택한다.

## 변경

- 24-D residual 범위, terrain_mid scale, 경사면 body/stride/clearance authority는 유지했다.
- adaptive 종료 보상: 완주 +120, no-progress -60, 물리 실패 -80, 시간 초과 -40.
  물리 실패가 no-progress와 겹치면 물리 실패만 적용한다. 기존 dense progress/velocity/
  efficiency shaping은 유지한다. legacy reward clip ±50 뒤에서 종료 보상을 더하므로
  +120이 +50으로 잘리지 않는다. legacy 18-D 환경의 보상 상수는 변경하지 않았다.
- episode-length에 도달해 미완주이면 명시적 timeout 종료와 페널티를 기록한다.
  baseline 비교나 영상의 짧은 재생 horizon은 환경 episode-length와 다르며, 이 재생
  길이에 도달했다는 이유로 timeout failure를 만들지 않는다.
- PPO discount 기본값 .99 → .997. 20ms tick에서 20초 뒤 보상의 할인 가중치는
  대략 .000043 → .0496이다. 먼 완주 결과를 더 반영하려는 설정이며 효과는 실측 전이다.
- best checkpoint는 **평가 완주율 우선, 같은 완주율이면 score-key(reward 기본) 우선**이다.
  cycle/best_score는 이 방식으로 선택된 checkpoint의 reward라 단조 증가를 보장하지 않는다.
  `cycle/best_success_rate`와 함께 본다. 보상 변경 전후 score를 직접 비교하지 않는다.
- 각 cycle best 영상은 기본 최대 40초. 조기 종료되면 짧아지고 영상 반복으로 채우지 않는다.
- `videos/best.termination.json`의 diagnostic_tail에 마지막 약 5초를 10Hz로 저장한다.
  요청/적용 action, 몸체 pose, 관절, 명령, raw/confirmed contact, scheduler 상태,
  보폭/phase duration, posture/height, clearance/apex/transfer, latched target,
  candidate rejection status, support margin, HOLD/종료 지표를 포함한다.
  이 파일은 기존 cycle best artifact에 포함된다. PPO 평가 episode 자체의 trace가 아니라
  best 영상 재생 seed의 trace라는 점을 구분한다.
- retry 소진 후 기본 advance. 실패 기록/통과 여부를 유지하며 다음 지형 시도를 성공으로
  표시하지 않는다. 프로세스 crash나 checkpoint 누락은 정상적인 stage 미통과와 달라 중단한다.

reward contract는 `adaptive_completion_outcome_v5`. 관측/network/action 크기는 동일하다.
수치들은 최초 조정값이며 학습 성공을 확인한 값이 아니다. 코드 문법/source hash만 확인하고
PPO, 시뮬레이션, renderer 및 unit test 실행은 사용자에게 맡긴다.

## 기존 경사면 best에서 재개

아래 checkpoint의 source revision은 `6cba85352f07f76b8d5e07e2863b2ec1d201f5ec`다.
`--migrate-completion-reward`는 해당 revision의 검토된 reward/metadata 파일 hash만 이전
허용하며 다른 controller/map/network hash 변경은 여전히 거부한다. old score는 이전하지 않는다.
new run과 새 evaluation으로 best를 다시 선택한다. flag는 첫 cycle에만 전달된다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/train_adaptive_curriculum.sh \
  --profile teacher --perception teacher --start-index 1 \
  --run-name adaptive-gt-completion-resume \
  --restore /home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum/adaptive-grid-v5-gt-teacher/01_tripod-ramp8_try01/checkpoints/000000614400 \
  --migrate-completion-reward \
  --timesteps-per-stage 800000 \
  --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 5 --num-eval-envs 8 --episode-length 8000 \
  --action-profile terrain_mid --discounting .997 \
  --baseline-comparison-seconds 20 --best-video-duration 40 \
  --max-retries 1 --on-stage-failure advance \
  --wandb --wandb-mode online
```

경사면을 이번에 다시 시도하고, 최초 실행+재시도 1회 후 미통과여도 낮은 계단으로 간다.
즉시 다음 지형부터 시작하려면 `--start-index 2`로 바꾼다. 새 reward의 checkpoint에서
이어서 시작할 때는 `--migrate-completion-reward`를 빼고 해당 checkpoint를 restore한다.
사용자가 기존 학습 프로세스를 별도로 실행 중이라면 그 프로세스의 현재 Python 모듈은
자동 갱신되지 않는다. 새로운 명령에서만 이 변경을 적용한다.

## 확인할 지표

`cycle/best_success_rate`, `eval/episode_termination/no_progress`,
`eval/episode_termination/timeout`, `eval/episode_reward/success`,
`eval/episode_reward/failure`, `eval/episode_reward/no_progress`를 확인한다.
comparison에서 baseline만 잘 걸으면 diagnostic_tail의 applied residual, phase freeze,
contact recovery를 확인한다. base/policy 비교는 두 seed의 표본이므로 완주율 추정 자체와
동일한 통계로 해석하지 않는다. 전체 curriculum 누적 best 영상은 추가하지 않는다.
