> 현재 권장: **v6 수정 + 기존 계단 가중치**는
> `bash /home/huro/Hexapod-Robot-integration/scripts/resume_stair5_path_v6.sh`를 사용한다.
> 아래의 v5 전용 재개/이전 미지원 설명은 이전 상태다. `--migrate-path-v6`가
> d0370b3의 terrain v5 source 전체 hash를 확인한 뒤 명시적으로 이전한다.
> 이전 가중치의 원본 파일은 변경하지 않는다. 실행/학습 검증은 아직 하지 않았다.

# 매 평가 best 영상과 계단 checkpoint 재개

`--best-video`는 이제 각 학습 후 평가 종료마다 현재 cycle의 best-so-far 영상을 게시한다.
204800 / 409600 / 614400 / 819200의 평가가 있으면 네 번 게시한다.
NEW BEST가 아니어도 같은 best 가중치로 다시 생성해 평가 시점과 best 시점을 각각 기록한다.
초기 step 0는 제외하며 cycle 종료에 다섯 번째 중복 영상은 생성하지 않는다.
선정 기준은 기존처럼 완주율 우선, 동률이면 reward다.

각 영상은 `videos/eval_<evaluation_step>_best_<best_step>.gif`에 저장한다.
`videos/best.gif`는 마지막 게시 영상의 복사본이며 manager와의 호환 경로다.
`monitor/eval_video_<evaluation_step>.json`에 checkpoint/score/영상 연결을 남긴다.
W&B `cycle/best_video`, `cycle/video_evaluation_step`, `cycle/best_step`으로 확인한다.
Best가 같으면 영상 내용도 같을 수 있다. 새 정책의 매 평가 영상과는 다르다.
렌더링은 callback에서 수행하므로 그동안 다음 PPO 업데이트가 기다린다.
업로드는 W&B에 enqueue하며 네트워크 상황에 따라 표시가 지연될 수 있다.
실패는 status/error로 남기며 PPO를 중단하지 않는다. 무손실 업로드를 보장하지 않는다.
Baseline 비교는 기존대로 cycle 종료에 한 번만 수행한다.

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
