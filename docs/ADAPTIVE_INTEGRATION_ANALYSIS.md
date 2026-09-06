# Adaptive integration 분석 및 결정

2026-09-07. main source: `9752c760002ac2dc7cc493ed634a881f87b7fd1e`.
실제 base는 후속 URDF-only commit `de77791d6386309295be5c8d173a1f51eca99930`이다.
두 revision 사이 STM32 변경은 없다. adaptive source는 `f9d80587ff086a1f3df51b81185e742094c109c9`.
전체 merge 없이 main에서 `codex/adaptive-hybrid-rl-integration`을 만들었다.

## 변경 전 분석

- main RL은 session / observation history 32 / age 60 ms / command timeout 100 ms /
  wrap-aware sequence / plan ID+mask 검증을 갖췄다. Workspace의 9-point preflight 후
  FootTrajectory pending→active latch가 있다. 입력 RobotRlAction은 absolute posture와 dx/dy/dz/dh다.
- main manual은 phase 1초, cubic XY+parabolic Z+radial 7 cm, swing 20 cm다.
  Wave 0,5,1,3,2,4; speed .2; stance 5 phase; 첫 순회 half stance를 사용한다.
  Early .5, Late .12 m/s·inward .096 m/s·10 cm 한계, support recovery와 전원 안전이 우선이다.
- MJX adaptive v3는 action24, actor4410/critic4725. wide25 후보를 local projection에도
  재사용해서 X를 decoder에서 늘려도 실제 ±6 cm endpoint가 없다. Tripod .5초도 main과 다르다.
- correction body_offset는 사용자 속도의 적분 translation이다. adaptive body height는
  phase별 absolute offset이므로 별도 state와 rate/workspace gate가 필요하다.
- main SPI v2는 raw 24-byte payload 보관까지이며 RL API와 연결되지 않았다.
  Jetson 폴더에는 실행 코드가 없다. ISR은 완료 flag만 기록해야 한다.

## 통합 결정

1. main의 HW/STM32를 유지하고 adaptive root mjx subtree(legacy replay 의존성·가중치 포함),
   viewer/train wrapper 3개와 관련 docs 4개만 선택적으로 가져온다. SW/mjx·다른 firmware는 가져오지 않는다.
2. PolicyAction24 v4와 final AdaptiveExecutionPlan을 분리한다. geometry가 Z를 소유한다.
3. wide ±8 cm grid와 local X ±6/Y ±4 cm grid를 분리한다. map 5 cm는 유지한다.
4. adaptive timing/trajectory 상수를 양쪽에 명시하고 golden tests로 drift를 검출한다.
5. final target frame, body offset, gait negotiation, freshness/latch를 내부 API에서 먼저 고정한다.
6. SPI는 마지막에 128-byte explicit little-endian fixed-point/CRC로 연결한다.
7. manual/legacy trajectory와 안전 상태기는 유지한다. 새 기능은 adaptive 실행에서만 선택한다.

빌드·단위 계약 확인은 최소 범위로 진행한다. Stage0 viewer, PPO, 보드 플래시와 물리 동작은
사용자 검증이며 코드 구현과 구분한다. 전체 변경 파일과 검증 결과는 최종 통합 보고서에 기록한다.
