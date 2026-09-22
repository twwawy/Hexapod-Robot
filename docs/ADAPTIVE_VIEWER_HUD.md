# Hybrid foothold viewer

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
bash scripts/view_foothold_planner.sh \
  --controller adaptive --terrain steps --perception lidar \
  --gait-mode hybrid --speed 0.04
```

이 명령은 학습 가중치 없이 LiDAR와 geometry/controller로 계획한다.
`--stage0`는 추가 GT 비교 진단이며 시각화를 켜기 위한 필수 옵션이 아니다.
GT 비교 metric을 보고 싶을 때만 추가한다. 첫 JIT 및 이후 실행 비용은 남아 있다.

HUD: Actual gait(실제 Tripod/Wave), Supervisor request(normal/short/Wave/HOLD),
contact/phase 대기 상태, 적용 stride/period, map known, 마지막 실제 gait 전환.
각 다리의 STANCE/SWING/LATE/TOUCHDOWN/HOLD와 confirmed contact, safe candidate 수를 표시한다.
화면은 ASCII로 표시해 글꼴 호환 문제를 피한다.

기본: 청록 grid + 주황 proposed landing + 빨강 latched target.
기본 후보/FOV/세부 reference label은 숨긴다.

| 키 | 기능 |
|---|---|
| N | 후보 표시 |
| B | 거절 후보 표시 및 후보 레이어 켜기 |
| V | nominal/reference/RL request 세부 표시 |
| G | LiDAR FOV |
| M | 지도 |
| W/S, 위/아래 | 전후 속도 |
| A/D, 좌/우 | yaw |
| Space | 명령 정지 |
| Enter | pause |
| H | reset(속도도 0으로 초기화) |
| P | plan/map/trace 저장 |

Hybrid는 deterministic supervisor가 조건에 따라 전환한다. Tripod가 가능한 지형에서
Wave 사용을 강제하지 않는다. 실제 전환은 접촉/phase 경계 조건을 따라 요청보다 늦을 수 있다.
HOLD 사유는 HUD의 요약과 후보별 콘솔 진단을 함께 확인한다.
강제 전환 키를 추가하지 않았다. 렌더링·시뮬레이션은 실행하지 않았고 Python 문법만 확인했다.
