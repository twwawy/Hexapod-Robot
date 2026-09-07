# Elevation grid 기반 residual RL — 관측/network v5

## PPO 초기화 shape 오류 수정

Brax PPO는 관측 크기를 `{'state': (7890,), ...}`로 전달하지만 초기 CNN 코드는
정수 크기만 가정해 `(1, (7890,))` 배열을 만들려 했다. actor/critic 모두 정수,
1-D tuple, JSON list를 정수 폭으로 정규화하도록 수정했다. 동작/관측/network 차원은
변경하지 않았다. 해당 오류는 네트워크 초기화 단계에서 발생했으므로 그 실패 run에는
복원할 학습 checkpoint가 없다. 아래 학습 명령에서 run-name만
`adaptive-grid-v5-lidar-mid-shapefix`로 변경하고 restore 없이 다시 실행한다.
회귀 테스트는 추가했으며 실행은 사용자에게 맡긴다.

## 구현 범위와 실행 상태

`codex/adaptive-hybrid-rl-integration`에서 수정했다. 이번 변경의 simulation,
PPO, unit test, hardware 실행은 모두 사용자에게 맡긴다. 보행 성공을 확인한 버전이라는
의미가 아니다. 기존 학습 프로세스를 시작하거나 종료하지 않는다.

물리적 action은 `adaptive_hybrid_geometry_residual_24_v4`를 유지한다.
관측은 `adaptive_hybrid_elevation_grid24x24x6_v5`, network는
`elevation_cnn_16_32_dense64_v1`로 분리했다. 기존 v4 checkpoint는 호환 오류로
거부한다. `--restore`나 `--migrate-flat-boxes`로 이전 네트워크를 우회 로드하지 않는다.
기존 run 기록/가중치를 삭제하거나 덮어쓰지 않는다. 진행 중인 이전 학습은 새 구조로
전환되지 않으며, 이전 checkpoint에서 다음 subprocess를 시작하면 계약 불일치가 난다.
이전 run을 이어갈 때는 그 checkpoint에 기록된 source revision을 별도 checkout해야 한다.

## 데이터 흐름

```text
LiDAR returns → 64×64 rolling elevation map (5 cm)
                      ├→ wide/local foothold geometry → safe reference
                      └→ body-yaw-aligned 24×24×6 local grid → CNN
robot/gait/contact/candidate vector ────────────────────────────┤
                                                              ↓
                                                         24-D residual
                                                              ↓
                          feasible stride / landing / clearance projection
                                                              ↓
                                            latched classical gait + safety
```

LiDAR 모드에서는 actor grid와 foothold geometry 모두 센서 지도를 사용한다.
teacher/oracle은 명시적으로 GT를 사용하며 LiDAR 학습 결과와 구분한다.

로컬 grid는 몸체를 중심으로 앞/왼쪽 방향에 정렬된 약 1.2 m 정사각형이다.
6개 채널은 몸체 기준 상대 높이, known, confidence, age, 전방 slope, 좌측 slope다.
unknown의 높이/slope는 0으로 mask하고 known/confidence=0, age=1을 전달한다.
이는 unknown을 안전한 평면으로 판정하는 동작이 아니다.
confidence는 관측 age와 최근 spread의 휴리스틱이며 보정된 확률값은 아니다.

CNN: Conv 3×3 stride 2, 16 channels → Conv 3×3 stride 2, 32 channels → flatten → Dense 64.
이 latent와 기존 4,434-D vector를 결합해 256/256/128 MLP로 정책을 만든다.
actor 입력은 **7,890-D**, critic은 **8,205-D**다. actor와 critic encoder는 독립적이다.
grid의 물리적 채널 scaling은 유지하고 기존 vector는 PPO running normalization을 사용한다.
normalizer 저장 형식은 전체 observation을 유지하므로 메모리/처리량은 실측이 필요하다.

Viewer의 M 지도 표시는 점 대신 5 cm 반투명 사각 셀로 그린다.
이 시각화 변경 때문에 viewer를 자동 실행하지 않는다.

## 불필요한 정지 수정

1. **셀 lifetime min/max 누적 제거**: 최신 scan 상단 높이를 대표값으로 채택한다.
   spread는 현재 scan 범위, 직전 scan과의 높이 변화, 시간상수 2초로 감쇠한 과거 spread의
   최댓값이다. 반복되는 실제 단차는 유지하고 오래된 단일 극값은 감쇠한다.
   미관측 셀의 값/타임스탬프는 갱신하지 않는다. MJX와 Jetson은 같은 fusion 함수를 사용한다.
   이것은 robust quantile/다중 surface estimator가 아니며 센서 노이즈에 대한 보정은 남아 있다.
2. **보폭은 선호값**: 요청 이하에 safe stride가 있으면 그중 가장 큰 값을 사용한다.
   없으면 전체 feasible set에서 요청에 가장 가까운 값을 선택한다. 안전한 stride가 있는데
   RL 요청보다 크다는 이유만으로 HOLD하지 않는다. Tripod oversize 1.3은 normal 1.0이
   가능한 경우에만 허용한다. Wave도 동일한 preference projection을 사용한다.
3. **초기 classical handoff**: 한 번 feasible이면 영구 종료하던 조건을 바꿨다.
   normal stride의 현재/다음 Tripod phase가 0.4초 연속 feasible이고 all-contact phase
   boundary에 도달해야 handoff한다. 초기 classical은 기존처럼 stride 0.5, 최대 24 phases다.
   nominal 착지 후보에서 관측된 unsafe terrain은 bootstrap으로 무시하지 않는다.
   fixed Wave stage에서는 Tripod bootstrap을 사용하지 않는다.
4. **정지 원인 기록**: 아래 W&B 지표를 추가했다. 기존 cycle best 영상/score 업로드 경로는 유지한다.

이 변경은 무조건 전진하는 blind controller가 아니다. handoff 이후 완전히 unknown인
다음 착지에 대한 지속적인 contact-inferred creep는 아직 구현하지 않았다.
UNKNOWN→HOLD, 접촉 확인, IK/workspace, path collision, 지지 여유 12mm는 유지한다.
coverage 60%, plane RMS 8mm도 이번에는 임의로 낮추지 않았다.

예측 CoM만 앞으로 옮기는 수정은 적용하지 않았다. 현재 실행 endpoint는 body-frame latch라
CoM만 이동하면 landing frame과 불일치한다. 이후 touchdown 시점 body motion과 landing target을
같은 frame/time으로 모델링한 뒤 support 검사와 map query를 함께 수정해야 한다.

## W&B에서 확인할 것

episode 합산 값은 아래 이름 앞에 `eval/episode_`가 붙는다. 각 tick에 초 단위 dt를
기록하므로 평가 결과는 episode당 평균 대기 시간(초)이다. 원인별 증거는 서로 겹칠 수 있다.
단일 원인으로 확정하는 counter가 아니며, 모든 contact/scheduler freeze를 포괄하지는 않는다.

| 지표 | 의미 |
|---|---|
| `hold_planner_s` | 이동 명령 중 다음 phase가 planner permit을 받지 못한 시간 |
| `hold_contact_wait_s` | 다음 phase 대기 중 all-contact가 아닌 시간 |
| `hold_map_unknown_s` | HOLD 중 active leg의 support coverage 후보가 부족한 시간 |
| `hold_surface_rejected_s` | HOLD 중 active leg 후보에 rough/edge가 관측된 시간 |
| `hold_ik_rejected_s` | HOLD 중 terrain+IK 통과 후보가 없는 active leg가 있는 시간 |
| `hold_path_rejected_s` | HOLD 중 terrain+IK+path+coverage 후보가 부족한 시간 |
| `hold_support_rejected_s` | HOLD 중 선택된 계획의 지지 여유가 12mm 미만인 시간 |
| `bootstrap_classical_s` | 초기 classical proposal이 활성인 시간; 실제 이동 시간과 다름 |
| `stride_preference_projected_s` | feasible projection이 RL 요청보다 큰 stride를 선택한 시간 |

영상에서 정지가 줄었는지와 함께 success, slip, touchdown error, projection, scheduler fault를
확인한다. reward 상승만으로 지형 적응 성공이라고 판정하지 않는다.
3초 no-progress watchdog은 유지했다. 정지 종료를 늦춰 성공처럼 보이게 하지 않는다.

## 사용자 실행 명령

새 network이므로 **restore 없이 새 run**으로 시작한다. 아래 명령은 GUI를 열지 않으며,
기존처럼 cycle별 best 영상과 score를 W&B로 보낸다. 전체 best 영상 추가 저장은 하지 않는다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/train_adaptive_curriculum.sh \
  --profile full --perception lidar \
  --run-name adaptive-grid-v5-lidar-mid \
  --timesteps-per-stage 800000 \
  --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 5 --num-eval-envs 8 --episode-length 8000 \
  --action-profile terrain_mid --best-video-duration 12 \
  --max-retries 3 --on-stage-failure advance \
  --wandb --wandb-mode online
```

`advance`는 실패한 stage도 표시하고 다음 지형을 시도한다. 통과를 의미하지 않는다.
통과한 stage만 진행하려면 `--on-stage-failure stop`을 사용한다.
메모리가 부족하면 `--num-envs 256`으로 낮춘다. 이번 변경의 GPU 사용량/속도는 측정하지 않았다.

사용자용 unit checks (rollout/PPO 없음):

```bash
python -m unittest mjx.tests.test_adaptive_grid mjx.tests.test_adaptive_hybrid mjx.tests.test_adaptive_v4
```

새 tests는 오래된 outlier 감쇠/실제 반복 단차 유지, 미관측 셀 age 보존, grid yaw/mask,
stride fallback, CNN batch shape와 grid gradient를 확인한다. 이번 작업에서는 실행하지 않았다.
