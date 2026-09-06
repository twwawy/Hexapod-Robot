# stage31 level6 학습 정책 실행

선택한 run은 `progress-v2-stage31-level6_20260828-111825_seed40`이다.
로컬 `mjx/runs/terrain/`에 있던 실제 Brax/Orbax checkpoint를 이 패키지에 포함했다.
W&B 로그나 GIF를 정책으로 사용하는 것이 아니라, 저장된 가중치와 관측 정규화 통계를 복원한다.

**착지점 뷰어 안에서 이 정책을 쓰는 기본 실행 명령:**

```bash
cd /home/huro/Hexapod-Robot
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/view_foothold_planner.sh --terrain steps
```

이 명령은 기존 넓은 코스·LiDAR 높이 지도·착지 후보 화면 안에서 stage31 정책으로 움직인다.
자세한 조작법과 관측/제어의 구분은 [착지점 뷰어 안내](../../../docs/HEXAPOD_FOOTHOLD_PREVIEW_USAGE.md)를 따른다.
아래는 별도 `view_trained_policy.sh`의 run 지형 재현 모드 설명이다.

```bash
cd /home/huro/Hexapod-Robot
bash scripts/view_trained_policy.sh
```

가상환경은 `/home/huro/.venvs/hexapod-mjx`이다. 스크립트가 직접 사용하므로 activate는
선택 사항이다. 기존 JAX/MJX/Brax 학습 환경이 필요하며, 화면 표시만 가능한 MuJoCo 환경으로는
실행할 수 없다. 첫 실행은 JAX 컴파일로 시간이 걸릴 수 있다.

| 항목 | 값 |
|---|---|
| 선택 기준 | 해당 run의 `monitor/level_best_checkpoint.json` |
| checkpoint step | **1,703,936** (`000001703936`) |
| 기록된 평가 점수 | 171.364044; 저장된 점수이며 이번에 재평가하지 않음 |
| W&B project / run ID | `hexapod-real` / `9w1fzji2` |
| 입력 | 146-D observation + 저장된 정규화 통계 |
| 출력 | 18-D adaptive swing residual v3 |
| 네트워크 | 256 → 256 → 128, SiLU, deterministic inference |
| 지형 | 7단 × 6.5 cm, 총 높이 45.5 cm |
| 제어/물리 주기 | 20 ms / 2.5 ms |
| 기본 reset seed | 20040; best 영상 생성 코드의 training seed 40 + 20000 |
| 원본 코드 기준 | 기록된 Git revision `0805164a5db4e85f1039637c829b760e5bc3d87b` |

이 checkpoint는 level 최고 점수 기준이다. 기존 학습의 `best_safe` 기준 통과 모델을
선택했다는 뜻은 아니다. 선택 당시 평가 기록은 `recorded_score.json`에 그대로 포함한다.

## 구성과 조작

시뮬레이터 상태·지형 관측 → 저장된 정규화 → PPO 정책 → STM32 기반 gait/잔차 제한/IK →
MuJoCo MJX 동역학 순서로 진행한다. 박스 몸체·캡슐 링크·구형 발을 사용하는 학습 환경이다.
현재 `view_foothold_planner.sh`의 LiDAR 높이 지도·착지점 보정은 이 경로에 연결하지 않았다.
이 정책이 학습한 terrain 관측은 시뮬레이터에서 제공한다.

| 키 | 동작 |
|---|---|
| Space | 시뮬레이션 일시정지 / 재개 |
| R | 같은 seed와 원래 샘플된 command로 reset 후 실행 |
| ↑ / ↓ | 전진 command ±0.02 m/s, 범위 0~0.12 |
| ← / → | yaw command ±0.1 rad/s, 범위 ±0.3 |
| F | 로봇 추적 카메라 전환 |

처음에는 키 입력 없이 저장 설정으로 동작을 본다. 방향키를 누르면 command를 변경한
실험이 된다. 이 run의 학습 yaw 범위는 0이므로 회전 결과는 학습 범위 밖의 동작이다.
종료 조건이나 episode 길이에 도달하면 마지막 자세를 유지하고 원인을 표시한다. R로 다시 시작한다.
HUD의 `action norm`은 복원된 PPO가 출력한 action 크기다.

```bash
# 평지에서 정책 동작 비교
bash scripts/view_trained_policy.sh --terrain flat

# 정지 화면에서 시작하거나 재생 속도 조절
bash scripts/view_trained_policy.sh --paused --speed 0.5

# 다른 초기 난수 상태
bash scripts/view_trained_policy.sh --seed 40
```

## 코드와 기록의 차이

현재 작업 폴더의 v4 정책 계약을 이 checkpoint에 적용하지 않는다. 기록된 커밋의 코드와
URDF를 `mjx/generated/trained_policy/source-0805164a5db4/`에 추출해 별도로 import한다.
원본 커밋의 level6 계단은 10 cm지만 run 기록은 6.5 cm이므로, 저장된 지형 치수로
해당 level 정의를 대체한다. 나머지 설정은 `environment.json`에서 읽는다.

학습 당시 미커밋 소스 전체는 run에 보관되지 않았다. 기록된 reward 계약과 커밋의
reward 코드에도 차이가 있어, 이 실행은 **가중치와 기록 기반 환경의 복원**이다.
W&B 영상·점수·종료 시점의 완전한 재현을 확인한 것은 아니다.

현재 패키지 파일의 해시 확인과 소스 추출까지만 수행했다. 사용자 요청에 따라 가중치 로딩,
정책 추론, GUI 실행, 동역학 보행 검증은 수행하지 않았다.

`manifest.json`에는 출처·checkpoint·SHA-256을 기록했다. 실행 시의 소스 경로·seed·환경은
`mjx/generated/trained_policy/replay_manifest.json`에 기록한다. 이 패키지에는 새 학습이나
W&B 업로드를 수행하는 코드가 없다.
