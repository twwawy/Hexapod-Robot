> 이 문서는 이전 개발 단계의 설계/사용 이력을 포함합니다. 현재 v4 contract, timing, 실행 경로는 [통합 문서](ADAPTIVE_INTEGRATION_V4.md)를 따릅니다.

# 구현 전 분석: LiDAR foothold / Tripod–Wave supervisor

기준: 작업 브랜치 `3c8f528`, 원격 main `90a8950` (2026-09-06 fetch).
분석을 사용자에게 먼저 보고한 뒤 구현한다. 이전 요청으로 진행 중이던 25개 후보/oracle 변경은
미커밋 초안이었으며 실행 검증하지 않았다. GUI/보행/PPO 검증은 사용자에게 맡긴다.

## 기존 데이터 흐름과 gate

JAX MID-360 ray → 64×64/5 cm/60초 지도 → nominal 주변 3×3/±4 cm 후보 →
중심+네 이웃 all-known → spread 2.5 cm → XY 요청과 가까운 후보 → terrain Z+
32 mm 발 반지름 → clearance → tripod swing 경계 latch → posture/contact/IK.

후보가 사라질 수 있는 지점은 FOV·가림·dropout·지도 범위·age·중심/주변 관측·spread·
검색 범위·경로 IK·clearance·tripod 전체 거부·정지/스윙 경계 대기다.
점군만 보고 실제 원인을 확정할 수 없다. 기존 selected 이전 탈락 진단이 없어 원인 분리가 어려웠다.

## 기존 action / observation

23-D: 0:12 nominal 기준 landing XY, 12:18 clearance, 18 pitch, 19 roll, 20 height,
21 stride, 22 phase duration. 새 제안과 roll/pitch 순서가 다르다.
XY/clearance/stride/duration은 경계에서 고정하고 자세 목표는 경계 수락 후 rate limit한다.

641-D actor = 155-D 상태 + 6×9×9 후보.
상태: command 5, velocity/gravity 9, joints 36, feet 18, contacts 6, phase/leg state 12,
twist 4, IK/limit 12, accepted/previous action 46, posture 3, stride/duration/reject/height 4.
critic 764-D에는 후보 GT/error 108과 지형 15가 추가된다.
normal, support margin, gait mode, maximum feasible stride는 없다.

## main의 Wave semantics

근거 파일은 `SW/STM32/WAVE_GAIT.md`, `gait_manager.c`, `foot_trajectory.c`,
`workspace_limiter.c`, `robot_config.h`다. C 전체를 복사하지 않는다.

- 순서 RF → LB → RM → LF → RB → LM (indices 0,5,1,3,2,4).
- 한 발 1초; Wave speed scale .2; 다섯 stance 위상의 누적 이동.
- 첫 여섯 위상 stance 이동 반감, 실제 발 memory부터 연속 시작.
- swing 완료 + 6개 confirmed contacts 뒤 패턴 전환; 시작 안정 대기 100 ms와 새 preview.
- raw touchdown 후보에서 고정, confirmed early landing, late 탐색 100 mm 한계.
- 지지발 접촉이 사라지면 위상 정지/재접촉 탐색; late 동안 다른 stance 이동 정지.
- Wave에서는 tripod 공통 Z 복구 해제.
- Tripod 입력은 두 위상 묶음, Wave는 매 발 갱신. 안전 supervisor는 무효한 다음 위상보다 HOLD를 우선한다.

MJX는 충돌 접촉을 사용하므로 하드웨어 FSR sampling/threshold를 재현했다고 주장하지 않는다.
기존 MJX Tripod .5초와 main 1초는 설정으로 분리한다. main의 swing 높이 .2 m까지 그대로
옮기지 않고 기존 MJX 기본 .06 m와 geometry-required clearance를 쓴다.

## 변경 설계

1. 25개 검색과 관측률/plane/edge/IK/path 진단, debug oracle.
2. geometry 모듈: normal/slope/confidence, 관측 support extent, IK/관절 margin,
   고정 top-K=3 조합과 support polygon; stride bank로 검증한 최대값.
3. 별도 scheduler/supervisor: NORMAL → SHORT → WAVE → HOLD.
   UNKNOWN으로 Wave에 진입하지 않는다. 복귀는 두 Tripod phase preview와 시간 hysteresis를 모두 요구한다.
4. 24-D: XY12, apex6, roll/pitch/height3, stride1, apex-phase1, transfer-timing1.
   duration은 controller가 계산; geometry 기준값 + residual + safety projection.
5. 관측/보상/단계별 학습 CLI, viewer와 회귀 테스트, 문서.

수정 대상은 adaptive controller/env/perception/policy/viewer/train과 신규 estimator/feasibility/
supervisor/scheduler 모듈이다. firmware 기본 모듈 및 stage31 replay는 보존한다.
새 adaptive 계약은 버전/차원을 바꾸고 이전 adaptive checkpoint를 거부한다.
최종 성공 조건은 사용자 GUI/보행 검증으로 판정하며 코드 작성만으로 충족됐다고 표시하지 않는다.

검증은 문법·CLI·정적 정합성으로 제한한다. 단위/회귀 테스트와 실행 명령은 제공하되 이 작업에서
장시간 JIT·동역학·학습을 자동 실행하지 않는다.
