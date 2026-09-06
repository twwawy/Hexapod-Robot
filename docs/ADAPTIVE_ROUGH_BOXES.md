# Rough 지형 지원 및 평지 checkpoint 이전

adaptive 환경에서 level 1/2를 막던 hfield를 8×4 고정 box pool로 교체했다.
기존 18-D 환경은 기본 `rough_boxes=False`로 heightfield 동작을 유지한다.
adaptive XML은 `generated/hexapod_adaptive_rl.xml`로 별도 저장한다.

box XY 크기/중심/높이는 `terrain_curriculum`의 기존 tile 정의를 사용한다.
물리 충돌과 LiDAR는 같은 geom을 사용하며 teacher/oracle/보상 GT는 그 box
영역의 상면 높이를 조회한다. 타일 경계에서는 접하는 box 중 높은 상면을 사용한다.
이 지형은 기존 보간 heightfield와 달리 타일 사이 수직 단차가 있다.
map이나 후보 수를 변경하지 않고, 학습 step 내부에는 고정 shape JAX 연산만 사용한다.

curriculum 시작 시 전체 stage의 지형 종류/번호/gait를 정적으로 확인한다.
이는 실제 scene 생성이나 물리 실행 검증을 대신하지 않는다.

## 기존 평지에서 재개

`--migrate-flat-boxes`는 `--restore`와 함께 첫 cycle에만 전달된다.
flat checkpoint의 action/observation/network 크기를 검사하고,
controller·IK·perception 등 변경을 허용하지 않는다.
이전이 허용되는 네 파일의 과거 hash는 저장소 `786ff09` 내용과 일치해야 한다.
허용 파일: `adaptive_gait_env.py`, `rough_terrain_env.py`, `prepare_rl_scene.py`,
`adaptive_gait_policy.py`. 임의 source mismatch를 무시하는 플래그가 아니다.
Replay는 계속 strict이며 이후 새 checkpoint는 새 소스 hash로 저장한다.
이전 내역은 새 metadata의 `explicit_migration`에 기록된다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/train_adaptive_curriculum.sh \
  --profile full --start-index 1 \
  --perception teacher --run-name adaptive-full-terrain-mid-boxes \
  --timesteps-per-stage 400000 \
  --num-envs 128 --batch-size 64 --num-minibatches 4 \
  --num-evals 5 --num-eval-envs 4 \
  --episode-length 8000 --action-profile terrain_mid \
  --best-video-duration 12 --max-retries 2 \
  --wandb --wandb-mode online \
  --migrate-flat-boxes \
  --restore /home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum/adaptive-full-terrain-mid/00_tripod-flat_try01/checkpoints/000000307200
```

平지 학습은 다시 하지 않고 index 1인 rough25부터 시작한다. 실제 checkpoint 경로는
사용자가 올린 로그 기준이다. 복원 실패 시 metadata를 덮어쓰지 말고 오류 내용을 확인한다.
cycle별 W&B score·영상·artifact 업로드를 유지한다.

episode 800은 16초로 rough 목표(약 2.31m)를 통과하기에 짧다.
위 8000은 최대 160초로 느린 Wave에도 시간을 주며 평가 비용은 증가한다.
rough Tripod만 할 경우 2000(40초) 등으로 줄일 수 있다.
목표 도달/실패 시에는 일찍 끝나며, 이 길이가 험지 완주를 보장하지 않는다.

이번 변경에서는 학습·시뮬레이션·렌더링·실기를 실행하지 않았다.
32개 box의 충돌 계산 비용 및 실제 이동 성능은 사용자 검증 대상이다.
