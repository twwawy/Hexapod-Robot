> 현재 권장: **v6 수정 + 기존 계단 가중치**는
> `bash /home/huro/Hexapod-Robot-integration/scripts/resume_stair5_path_v6.sh`를 사용한다.
> 아래의 v5 전용 재개/이전 미지원 설명은 이전 상태다. `--migrate-path-v6`가
> d0370b3의 terrain v5 source 전체 hash를 확인한 뒤 명시적으로 이전한다.
> 이전 가중치의 원본 파일은 변경하지 않는다. 실행/학습 검증은 아직 하지 않았다.

# 매 평가 현재 정책 영상

`--best-video`를 켜면 학습 후 평가마다 **현재 정책**을 렌더링한다.
W&B `cycle/current_video`, `cycle/current_score`, `cycle/current_step`을 확인한다.
Best는 완주율 우선, 동률이면 reward로 계속 보존한다.
Best가 갱신된 평가에서만 `cycle/best_video`와 best-policy artifact를 추가한다.
해당 평가의 현재 영상을 재사용하므로 rollout을 두 번 실행하지 않는다.
초기 step 0 및 cycle 종료 중복 영상은 제외한다.

현재 영상은 `videos/eval_<step>_current.gif`, 연결 정보는
`monitor/current_video_<step>.json`에 저장한다. 모든 평가 checkpoint는
current-policy artifact에 기록하며 best는 별도 best-policy artifact로 보관한다.
Best가 같더라도 current 영상은 새 checkpoint의 실제 정책을 보여준다.
동일 seed/명령 조건으로 비교하지만 단일 영상의 결과는 평가 성공률과 다를 수 있다.

재개 preset: 학습량 800000, 평가 설정 9, 평가 env 16, 영상 최대 20초,
추가 baseline 비교 0. Brax의 초기 평가를 제외한 학습 후 평가는 보통 8회이며
실제 간격은 PPO batch 단위로 결정된다(현재 구성에서 약 102400 step).
영상은 동기식으로 생성하므로 PPO가 기다리며, 평가 횟수 증가가 전체 학습을
더 빠르게 만든다는 의미는 아니다. 학습/영상 검증은 사용자에게 맡긴다.

## 지정 GIF의 checkpoint

`best_video_7_0e1017aebbbc13860373.gif`는 로컬 best.gif와 SHA256이 일치한다.
저장된 best pointer의 checkpoint는 다음이다.

```text
/home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum/forward-gt-terrain-mid/02_tripod-stair5_try02/checkpoints/000000204800
```

Recorded source: `d0370b3f30b0d3c83541c61a1df44bc87ebcf7d7`, terrain task, v5 관측.
저장된 모든 controller/source hash가 해당 revision과 일치하는 것을 파일 읽기로 확인했다.
v6는 입력 차원이 달라 직접 restore할 수 없다. 따라서 재개 브랜치
`codex/stair5-eval-video-resume`는 기록된 v5 소스에 영상 callback 변경만 선택 적용한다.
이 경로에는 v6 경로 bottleneck 보정이 없다. v6 자동 가중치 이전을 구현하지 않았다.

```bash
cd /home/huro/Hexapod-Robot-stair5-resume
source /home/huro/.venvs/hexapod-mjx/bin/activate
checkpoint=/home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum/forward-gt-terrain-mid/02_tripod-stair5_try02/checkpoints/000000204800
bash scripts/train_adaptive_curriculum.sh \
  --profile teacher --command-mode terrain --perception teacher \
  --start-index 2 --restore "$checkpoint" \
  --run-name stair5-best-eval-video-resume \
  --run-root /home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum \
  --timesteps-per-stage 800000 \
  --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 5 --num-eval-envs 16 --episode-length 8000 \
  --action-profile terrain_mid --discounting 0.997 \
  --best-video-duration 40 --baseline-comparison-seconds 20 \
  --promote-key eval/episode_terrain_success --promote-threshold 0.70 \
  --max-retries -1 --on-stage-failure stop \
  --wandb-project hexapod-forward-completion --wandb --wandb-mode online
```

Teacher curriculum index 2는 stage 1 Tripod, terrain level 5다.
이후 완주율 70%를 통과하면 다음 지형으로 진행한다.
학습/시뮬레이션/영상 생성은 실행하지 않았다. 문법과 source hash만 확인했다.
