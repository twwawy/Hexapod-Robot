# 통합 검증 기록

2026-09-22, macOS arm64. Python 3.13.12, JAX/jaxlib 0.6.2,
MuJoCo/mujoco-mjx 3.12.0, Brax 0.14.1. C native compiler는 Apple Clang,
ARM compiler는 STM32CubeIDE의 GNU Tools for STM32 14.3이다.

## 결과

| 검사 | 결과 |
|---|---|
| 추적 Python AST 및 shell syntax | 통과 |
| 원본 5개 tip이 HEAD의 조상인지 검사 | 5/5 통과 |
| CLI: 인자·공백·help·dry-run·잘못된 명령·원본 worktree·bundle 복원·덮어쓰기 차단 | 8/8 통과 |
| Adaptive CPU 계약 테스트 | 42/42 통과 |
| STM32 native 알고리즘 그룹 | 8/11 통과, 아래 기존 실패 3개 |
| ARM Cortex-M4 전체 compile/link | 통과, `Hexapod.elf` 생성 (flash 안 함) |
| 최신 main과 HW·STM32 Core/Drivers/.ioc 비교 | 동일 |

`./hexapod check --rl`은 모든 검사 그룹을 실행한 뒤 STM32 native 실패 때문에 **exit 1**을 반환한다.
실패를 skip하거나 성공으로 처리하지 않는다. 따라서 이 브랜치를 “모든 검사 통과”나
“실기 배포 검증 완료”로 해석하면 안 된다.

## 기존 STM32 실패 재현

원본 main `fdd7cfc8b98459905eebb3ee71009f8edd525b1e`의 Core/Drivers를 `git archive`로
별도 디렉터리에 복원한 뒤 **동일한 host runner**로 실행했다.
통합 트리와 원본 main 모두 아래 동일한 3개 그룹에서 실패했다.
이 비교는 통합 때문에 추가된 회귀와 원본/host 환경에서 재현되는 실패를 구분하며,
실제 보드에서도 같은 실패가 난다는 증명은 아니다.

- `workspace`: `WorkspaceTest_CheckReducedCommand`, `Core/Src/test/workspace_test.c:166`.
  최대 동시 입력이 반드시 즉시 거절되고 감속 preview로 들어간다는 기대가 맞지 않는다.
- `gait`: `GaitTest_CheckLimitedContinuousWalk`, `Core/Src/test/gait_test.c:1011`.
  최대 동시 입력의 원본 채택/최종 거부 조건에서 실패한다.
- `mode_transition`: `ModeTransitionTest_Run`, `Core/Src/test/mode_transition_test.c:514`.
  하위 모드 전환 검사의 합성 결과가 false다. 구체적인 보드 동작 원인은 미확정이다.

현장 보정값과 최신 동작을 되돌리거나 테스트 기대값을 완화하지 않았다.
원인 확정에는 최신 설정에 맞는 입력 조건과 모드 전환 요구사항 검토가 필요하다.
host runner는 실패 위치를 임시 복사본에만 삽입한다. 임베디드 테스트/제어 소스는 변경하지 않는다.
macOS에서는 HAL의 ELF RAM section 선언을 host 전용 preinclude로 제외하고 Mach-O dead-strip을 사용한다.
Linux의 기존 section garbage collection은 유지한다. 하드웨어 HAL 호출은 abort stub이다.

## 통합 중 수정한 실행 오류

`test_adaptive_v4.test_real_geometry_projection_and_z_ownership`이
`HardwareGeometry`의 `_config` 누락으로 실패했다.
공유 경로 검사에 필요한 `stair_clearance_extra=0.0` 기본값을 추가한 뒤 재실행하여 통과했다.
MuJoCo runtime과 MJX package 버전을 학습 요구사항에서 3.12.0으로 일치시켰다.

## 검사 범위

42개 RL 검사는 foot retry 5, grid 6, hybrid 16, C/Python v4 계약 5,
RC command 4, path bottleneck 4, migration layout 2개다.
전체 PPO 학습, GUI viewer, 실제 학습 checkpoint restore, Isaac Lab 실행,
실기 SPI/DMA/DRDY·모터·LiDAR/odom·계단 완주는 실행하지 않았다.
수치 계약/빌드 성공을 실제 보행 성능으로 대신하지 않는다.
