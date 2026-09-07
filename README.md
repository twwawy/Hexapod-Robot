# Hexapod Robot — RC 명령 추종 보행 학습

6개 다리·18개 관절을 사용하는 로봇의 MJX 기반 residual RL 프로젝트다.
현재 목표는 **조종기의 전진·후진·회전·정지 명령을 따라 안정적으로 걷는 정책**을 만드는 것이다.

작업 브랜치: `codex/adaptive-hybrid-rl-integration`

## 1. 현재 개발 방향

먼저 평지에서 조종 명령 추종을 학습한다. 학습 중에는 물리 조종기 대신
명령 생성기가 `[vx, wz]`를 제공한다. 실제 로봇에서는 조종기 입력을 이 경로에 연결하는 구조다.

| 구분 | 현재 내용 |
|---|---|
| 명령 | 전진·후진·제자리 좌/우 회전·전후진 곡선·정지 |
| 명령 변화 | 목표를 2~5초 유지하며 속도 변화율과 deadband 적용 |
| 학습 환경 | MuJoCo/MJX, 평지 RC curriculum |
| 지형 입력 | 아래 실행 명령은 `teacher`: GT 지형을 사용하는 모드 |
| 보행 단계 | Tripod → Wave → Hybrid |
| 승급 기준 | 평가 episode의 명령 추종 성공률 70% 이상 |
| 초기 가중치 | 기존 계단 checkpoint를 가져오지 않고 새로 시작 |
| 다음 개발 | RC 명령과 험지 목표를 함께 구성한 task, LiDAR 기반 정책 및 실기 연결 |

**현재 RC curriculum에 계단 등판은 포함되지 않는다.** 자유 회전·후진 명령과
일방향 계단 완주 목표를 함께 쓰려면 지형과 명령 분포를 추가로 설계해야 한다.

## 2. 제어 구조와 역할

```text
조종 명령 [vx, wz] + 로봇 상태 + 지형 정보
                    ↓
        Geometry planner / elevation grid
                    ↓
          Classical gait + 24-D RL residual
                    ↓
           Safety projection / gait supervisor
                    ↓
             발끝 목표 → IK → 관절 위치 목표
```

상위 입력은 속도 명령이지만 **다리·관절 실행은 위치 기반**이다.
RL은 보행 전략을 조절하고 geometry와 제어기가 실행 가능한 목표를 만든다.

| 소유 주체 | 역할 |
|---|---|
| Geometry | 지형 높이·표면·관측 신뢰도·안전한 착지 후보·경로 장애물 계산 |
| RL | 착지 XY, 발 높이, 몸체 자세/높이, 보폭, swing timing 조절 |
| Supervisor | Hybrid에서 Tripod normal → short-step → Wave → HOLD 결정 |
| 접촉·실행 제어기 | Early/Late Landing, 재접촉, 목표 latch, workspace/IK 검사 |
| STM32 실기 계층 | 실제 접촉 확인·gait 전환 시점·관절/PWM·Kill/Fault 안전 권한 |

착지 Z는 지형이 결정한다. Policy가 landing Z나 관절각/PWM을 직접 출력하지 않는다.
현재 MJX와 STM32의 구현 및 실제 동작 검증은 구분해서 관리한다.

### Policy 계약

Action은 `adaptive_hybrid_geometry_residual_24_v4`의 24-D다.
관측은 기존 상태·후보 feature에 24×24×6 elevation grid를 더해 CNN으로 처리한다.

| 인덱스 | 의미 | 기본 decoder 범위 |
|---|---|---|
| `0:12` | RF/RM/RB/LF/LM/LB 착지 XY residual | X ±6 cm, Y ±4 cm |
| `12:18` | 다리별 clearance residual | ±4 cm, geometry 필요 높이 이하로 제한 불가 |
| `18` | Roll residual | ±5° |
| `19` | Pitch residual | ±10° |
| `20` | Body height residual | ±3 cm |
| `21` | Stride 선호 | 0.5~1.3, feasible 영역으로 projection |
| `22` | Apex timing | 최고점 시점 조절 |
| `23` | XY transfer timing | 수평 이동 시점 조절 |

위 수치는 기본 decoder 범위다. 아래 명령의 `terrain_mid`는 action을 추가로 축소하며,
실제 착지와 궤적에는 geometry/safety 제한도 적용된다. Phase duration은 controller가 계산한다.

## 3. 새 RC 학습 시작

기존 로컬 환경 기준이다. **첫 실행에는 `--restore`를 넣지 않는다.**

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate

bash scripts/train_adaptive_curriculum.sh \
  --profile rc \
  --perception teacher \
  --run-name adaptive-rc-gt-flat \
  --timesteps-per-stage 800000 \
  --num-envs 512 \
  --batch-size 128 \
  --num-minibatches 4 \
  --num-evals 5 \
  --num-eval-envs 16 \
  --episode-length 2000 \
  --action-profile terrain_mid \
  --discounting 0.997 \
  --baseline-comparison-seconds 20 \
  --best-video-duration 40 \
  --promote-key eval/episode_command_success \
  --promote-threshold 0.70 \
  --max-retries -1 \
  --on-stage-failure stop \
  --wandb \
  --wandb-mode online
```

`--profile rc`가 RC command mode를 설정한다. 같은 RC run 안에서 retry 또는 다음 stage로
진행할 때는 manager가 checkpoint를 이어받는다. 다른 task/source의 checkpoint를
자동 호환으로 취급하지 않으며 계약 검사를 유지한다.

### 자주 바꾸는 옵션

| 옵션 | 조절하는 내용 |
|---|---|
| `--run-name` | 실험 폴더 및 W&B group 이름 |
| `--timesteps-per-stage` | stage의 각 학습 시도에 주는 학습량; PPO batch 단위로 실제 값이 달라질 수 있음 |
| `--num-envs` | 병렬 학습 환경 수 |
| `--batch-size`, `--num-minibatches` | PPO batch 구성; 환경 수와 함께 호환되는 조합 사용 |
| `--num-evals`, `--num-eval-envs` | 평가 횟수와 병렬 평가 환경 수 |
| `--episode-length` | episode 최대 길이; 현재 2000 step은 40초 |
| `--action-profile` | residual 허용 수준; 현재 `terrain_mid` 사용 |
| `--best-video-duration` | cycle best 영상의 최대 길이; episode 종료 시 더 짧아질 수 있음 |
| `--baseline-comparison-seconds` | cycle 종료 시 zero-action/policy 비교 길이 |
| `--promote-threshold` | 다음 stage로 넘어갈 평가 성공률 |
| `--max-retries` | `-1`이면 승급 기준을 통과할 때까지 무제한 재시도 |

## 4. 성공 기준과 W&B 확인

RC의 성공은 계단 완주가 아닌 **명령 추종**이다.

- Episode를 안전하게 마치고 명령 활성 시간의 70% 이상에서 선형 오차 ≤0.02 m/s,
  yaw 오차 ≤0.07 rad/s를 만족하면 해당 episode가 성공한다.
- Curriculum은 `eval/episode_command_success ≥ 0.70`일 때 승급한다.
  16개 평가 환경에서는 최소 12개 성공이 필요하다.
- 정지 명령에서 멈추는 것은 정상이다. 이동/회전 명령 중 멈추는 현상과 구분한다.
- 안전 실패 및 이동 명령 중 3초 no-progress 종료 조건은 유지한다.

여기서 **cycle은 `tryXX` 하나의 학습 시도**다.
평가 중 점수·성공률을 W&B에 기록하고, cycle 종료 시 그 cycle의 best checkpoint로
영상을 만들어 업로드한다. Best는 명령 성공률을 우선하고 동률이면 reward로 선택한다.
전체 curriculum의 누적 best 영상은 생성하지 않는다.

| 확인 항목 | 보는 이유 |
|---|---|
| `eval/episode_command_success` | 실제 승급 기준 |
| Cycle best score와 영상 | 같은 cycle에서 선택한 정책의 행동 확인 |
| 영상의 `vx`, yaw 명령 | 정상 정지와 명령 불이행 구분 |
| `cycle/best_video_status` | 영상 대기·렌더링·업로드 상태 또는 실패 확인 |
| Paired comparison의 선형/yaw 오차 | 같은 명령에서 zero-action 대비 변화 확인 |

로컬 결과는 `mjx/runs/adaptive-curriculum/<run-name>/` 아래 cycle별로 저장한다.
동일 이름의 폴더가 있으면 timestamp가 붙는다. 영상은 cycle 폴더의 `videos/best.gif`다.

## 5. 학습 중 소스 고정

위 shell 스크립트는 시작 commit을 `~/.cache/hexapod-training-sources/`의 독립 worktree에
고정한다. 모든 retry와 다음 stage가 같은 소스를 사용하므로 개발 저장소 수정이 섞이지 않는다.

- 시작 로그의 `PINNED SOURCE`, `SOURCE REVISION`을 보관한다.
- 실행 중인 고정 worktree는 수정하거나 삭제하지 않는다.
- 추적 파일에 미커밋 변경이 있으면 시작이 차단된다. 사용할 변경을 commit한 뒤 실행한다.
- Untracked 파일은 복사되지 않으므로 소스·모델 의존 파일도 commit되어 있어야 한다.
- 직접 Python trainer/manager를 실행하면 이 고정 기능이 적용되지 않는다.
- 이미 실행 중인 run에는 소급 적용되지 않는다.

## 6. 코드와 문서

| 경로 | 역할 |
|---|---|
| `scripts/train_adaptive_curriculum.sh` | 권장 학습 진입점 |
| `mjx/launch_adaptive_curriculum.py` | 실행 소스 고정 |
| `mjx/train_adaptive_curriculum.py` | stage·retry·승급 관리 |
| `mjx/train_adaptive_gait.py` | PPO·평가·W&B·cycle 영상 |
| `mjx/operator_commands.py` | RC 명령 생성·변화율·추종 기준 |
| `mjx/adaptive_gait_env.py` | 명령 조건부 환경·관측·보상 |
| `mjx/adaptive_gait_controller.py` | residual·swing·실행 목표 |
| `mjx/adaptive_gait_perception.py`, `mjx/adaptive_grid.py` | 지도 및 grid 관측 |
| `mjx/adaptive_foothold_estimator.py` | 안전한 착지 후보 계산 |
| `mjx/hybrid_gait_supervisor.py`, `mjx/wave_gait_scheduler.py` | gait 선택·접촉 단계 |
| `SW/STM32/` | 실기 제어기 |
| `SW/Jetson/` | 상위 실행 계획·통신 연결 |
| `HW/` | 기구·URDF·하드웨어 자료 |

현재 학습의 상세 명령 분포·보상·지표 해석은 [RC 명령 추종 학습](docs/ADAPTIVE_RC_TRAINING.md)에 정리했다.

## 7. 검증 범위

RC 학습 코드와 실행 경로를 구현한 상태다. RC 정책의 학습 성능, 실제 조종기 연결,
험지 조종 성능, STM32 실기 재현성은 별도 검증 대상이다.
학습·시뮬레이션·보드 실행은 사용자가 수행한다.

먼저 새 RC run에서 명령별 전진·후진·회전·정지와 W&B cycle 영상을 확인하고,
평지 명령 추종 성능을 확보한 뒤 험지 task 및 LiDAR/실기 연결로 확장한다.
