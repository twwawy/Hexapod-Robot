# Phase 경계 재접촉과 70% 통과 curriculum

## GT가 하는 일

teacher/oracle에서 terrain query는 GT 높이를 반환하고 known=100%로 만든다.
landing Z는 GT surface Z, swing은 경로 최고 지형과 시작/착지 표면의 높이를 비교해 계산한다.

```text
required = max(path_high - max(start_surface_z, landing_surface_z), 0)
clearance = clip(max(required + 0.06 + RL_clearance,
                     required + 0.02, 0.04), 0.04, 0.18)
```

clearance는 높은 쪽 endpoint 위로 추가로 드는 높이다. 착지 Z 자체가 상승하면 swing top도
따라 상승한다. 상한 18cm로 required+margin을 만족할 수 없으면 path 검사에서 거부한다.
지형 gradient는 body posture baseline, 상승 높이는 apex timing baseline에도 사용한다.
GT로 지형을 알아도 실제 발의 접촉 유지·재접촉 상태 기계는 별도로 동작해야 한다.
GT 계산이나 residual 범위는 이번에 바꾸지 않았다.

## 수정 범위

이번 수정은 MJX adaptive scheduler/controller와 curriculum이다. STM32 C 펌웨어는
변경하지 않았다. 새 boundary recovery가 실기에 포팅·검증됐다고 주장하지 않는다.
기존 18-D 경로 및 adaptive 24-D 입력/출력 차원은 유지한다. Scheduler 내부 상태는 늘었다.

- phase 완료 이력이 있고 이동 명령이 활성인 상태에서, 다음 phase 대기 중 발 접촉이
  끊기면 `recontact_active`로 진입한다. reset 직후에는 발을 임의로 내리지 않는다.
- 다음 swing의 planner permit과 복구 허가는 분리한다. 새 swing 불허여도 재접촉은 가능하다.
- 미접촉 발만 controller Z 방향으로 3cm/s 하강한다. 기존 Swing Late의 안쪽 이동은
  여기에 적용하지 않는다. 나머지 발 목표와 posture/height 적용값은 유지한다.
- raw contact 시 하강을 멈추고 confirmed contact를 기다린다.
- 최대 목표 하강 3cm 또는 1초. 한계 도달 시 fault/HOLD한다. IK/workspace가 막으면
  목표 적용을 취소하고 `recontact_ik_blocked`를 기록한다.
- 이동량은 IK 검사 후 채택된 발 목표 변화량이다. 실제 발이 이만큼 이동했다는 측정값은 아니다.
- 복구 중 phase/epoch를 진행하거나 gait를 바꾸지 않는다. all-contact가 0.1초 유지된 뒤
  복구를 끝내고, 유효한 현재 epoch 계획이 있을 때에만 다음 swing을 시작한다.
- raw/confirmed contact 샘플 주기와 기존 3초 no-progress watchdog은 변경하지 않았다.

W&B 지표: `recontact/active_s`, `recontact/exhausted`, `recontact/ik_blocked`,
`recontact/applied_descent_m`. exhausted/ik_blocked는 유지되는 flag이므로 episode 합은
이벤트 횟수가 아니다. 영상 sidecar의 scheduler에는 mask/time/distance/flag가 함께 저장된다.

## Curriculum

기본값은 `--promote-threshold .70 --max-retries -1 --on-stage-failure stop`.
`-1`은 통과까지 무제한 재시도다. 70% 미달을 자동 통과시키지 않는다. 프로세스 crash,
checkpoint 누락은 코드/실행 오류로 중단하며 무한 재실행하지 않는다. 완료된 cycle의
best로 다음 재시도를 이어가며 cycle별 W&B score/영상 저장은 유지한다.
`--max-retries 3 --on-stage-failure stop`처럼 유한 횟수로 바꿀 수도 있다.
무한 재시도와 advance를 함께 지정하면 모순된 설정으로 거부한다.

무한 재시도는 성공을 보장하지 않는다. 평가 환경 수가 작으면 seed에 따른 통과율 변동이
크다. 예를 들어 평가 8개에서 70% 문턱을 넘으려면 6/8=75%가 필요하다. 아래는 16개를 쓴다.
복구 한계 반복 등 제어기 문제는 재시도 횟수로 해결되지 않으므로 W&B 지표를 확인한다.

## 사용자 실행 — 경사면부터 재개

아래는 source 9e8c1ab의 completion-reward checkpoint를 명시적으로 이전한다.
`--migrate-recontact`는 검토된 controller/scheduler/env/metadata 파일 hash만 허용한다.
과거 코드의 점수를 새 제어기 성능으로 사용하지 않고 새로 평가한다. 새 버전 checkpoint로
재개할 때는 migration flag를 제거한다. 실제 학습·simulation·unit tests는 실행하지 않았다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/train_adaptive_curriculum.sh \
  --profile teacher --perception teacher --start-index 1 \
  --run-name adaptive-gt-recontact-until70 \
  --restore /home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum/adaptive-gt-completion-resume-20260907-233419-411263/01_tripod-ramp8_try01/checkpoints/000000614400 \
  --migrate-recontact \
  --timesteps-per-stage 800000 \
  --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 5 --num-eval-envs 16 --episode-length 8000 \
  --action-profile terrain_mid --discounting .997 \
  --baseline-comparison-seconds 20 --best-video-duration 40 \
  --promote-key eval/episode_terrain_success --promote-threshold .70 \
  --max-retries -1 --on-stage-failure stop \
  --wandb --wandb-mode online
```

사용자용 단위 검사: `python -m unittest mjx.tests.test_adaptive_hybrid`.
새 검사는 idle RF 미접촉 복구, raw 후보 정지, confirmed 안정 후 유효 계획으로만 launch,
수직 목표/다른 발 고정, 시간 한계 HOLD를 포함한다. 동역학/실기 검증을 대신하지 않는다.
