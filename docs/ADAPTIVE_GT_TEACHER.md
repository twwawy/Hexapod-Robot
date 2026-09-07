# GT teacher 우선 학습

LiDAR coverage와 RL을 동시에 해결하려고 반복 학습하는 대신, 정확한 지형으로 보행
전략을 먼저 학습한다. 새 `teacher` curriculum profile을 사용한다. 별도 Stage 0 프로세스는
필수가 아니며 cycle 끝의 짧은 baseline 비교를 진단에 사용한다. 이 변경이 제어기 문제를
해결했다는 뜻은 아니다. 이번 작업에서 simulation/PPO/비교 rollout을 실행하지 않았다.

## 실행

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/train_adaptive_curriculum.sh \
  --profile teacher --perception teacher \
  --run-name adaptive-grid-v5-gt-teacher \
  --timesteps-per-stage 800000 \
  --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 5 --num-eval-envs 8 --episode-length 8000 \
  --action-profile terrain_mid --baseline-comparison-seconds 20 \
  --best-video-duration 12 --max-retries 1 --on-stage-failure stop \
  --wandb --wandb-mode online
```

기존 LiDAR checkpoint를 restore하지 않는다. 초기 residual은 기존 terrain_mid의 고정
authority를 사용하며 자동 확대 스케줄은 추가하지 않았다. GT teacher의 actor grid와
planner는 GT를 사용한다. critic만 GT인 LiDAR student 학습과 다르다.
action/network contract는 이번 학습 orchestration 변경으로 바뀌지 않는다.

Tripod 순서: flat → ramp8 → stair5 → stair8 → rough25 → ramp15 → stair10 → rough50.
이후 기존 Wave/Hybrid 단계를 이어간다. 단위와 지형 정의는 terrain_curriculum.py를 따른다.
기존 fast/full/observe profile 순서는 유지한다.
통과 기준 미달 시 한 번 재시도한 뒤 정지한다. 실패한 기초 보행으로 어려운 지형에
계속 진입하지 않도록 추천 명령은 advance 대신 stop이다.

## 기본 보행과 정책 비교

매 cycle 종료 후 **그 cycle의 best checkpoint**와 **24-D zero-action**을 비교한다.
두 개의 같은 reset seed, 같은 환경/명령 sampler, 같은 최대 20초 horizon을 사용한다.
환경 종료 후 재시작하지 않고 결과를 고정한다. 20초에 끝나지 않으면 truncated로 표시하며
성공으로 간주하지 않는다. forward_m는 시작 몸체 전방으로 투영한 순변위여서 회전 명령이
있는 episode에서는 전체 경로 길이를 뜻하지 않는다.

이 비교는 학습 업데이트에 사용하지 않는다. PPO reward/best 선정, promotion의
`eval/episode_terrain_success`를 변경하지 않는다. 영상은 이전처럼 cycle best policy를
별도 영상 seed로 재생하며, paired 비교 seed의 영상이라고 주장하지 않는다.
전체 curriculum 누적 best 영상은 추가하지 않는다.

W&B 확인 항목:

- `comparison/baseline/forward_m`, `comparison/policy/forward_m`
- `comparison/*/hold_planner_s`, `comparison/*/hold_contact_wait_s`
- `comparison/*/success`, `comparison/*/no_progress`, `comparison/*/truncated`
- `comparison/policy_minus_baseline_forward_m`
- 기존 `cycle/best_score`, `cycle/best_video`, `cycle/best_video_status`

baseline과 policy가 모두 멈추면 planner/contact 문제를 먼저 확인한다. baseline만 잘 걸으면
residual/reward를 확인한다. 두 seed의 짧은 결과는 진단 표본이지 보행 성능의 확증이 아니다.
비교 실패는 `comparison/error`와 `monitor/baseline_comparison_error.json`에 기록하고
best 영상 생성은 계속 시도한다. 정상 결과 파일은 `monitor/baseline_comparison.json`이다.

`--baseline-comparison-seconds 0`이면 비교를 끈다. teacher profile 기본값은 20초,
기존 profile 기본값은 0이다. 비교에는 추가 JIT/rollout 비용이 들며 처리량은 측정하지 않았다.

GT teacher가 안정적으로 개선된 다음 LiDAR student 증류와 PPO fine-tuning을 진행한다.
이번 변경에는 자동 teacher-student 증류를 구현하지 않았다. 단순 입력 교체를 증류로 보지 않는다.
