# 발전 과정 및 통합 결정

2026-09-22 원격 branch 5개를 확인하여 `RL/unified-latest`로 통합했다.
최신 판단은 tip 날짜 하나가 아니라 공통 조상 이후의 기능·설정·정책 계약을 비교했다.
원본 SHA는 [integration-sources.json](integration-sources.json)에 고정했다.

## 브랜치별 처리

| 원본 | Tip | 원본 main 대비 | 통합 방식 |
|---|---|---|---|
| main | `fdd7cfc` (09-14) | 기준 | 최신 HW·STM32·캘리브레이션·센서·착지·64-byte 기본 SPI 보존 |
| codex/adaptive-hybrid-rl-integration | `1893969` (09-09) | 7 ahead / 10 behind | 실제 3-way merge: 신규 MJX 복구·HUD 적용, README는 최신 main에서 통합 안내로 갱신 |
| codex/cartesian-residual-rl | `f9d8058` (09-06) | 67 ahead / 92 behind | history-only merge: 현재 root mjx는 이미 선별 이식 후 발전한 코드. 원본 전체는 `history cartesian`으로 복원 |
| codex/stair5-eval-video-resume | `8948fc8` (09-07) | 2 ahead / 16 behind | history-only merge: 정확한 v5 소스는 `history stair-v5`로 복원. 현재 v6의 per-evaluation 영상 기능 유지 |
| codex/wave-gait-three-zone | `e035a3d` (09-06) | 0 ahead / 40 behind | 이미 main의 조상. 추가 병합 불필요 |

`history-only`는 `git merge -s ours --no-ff`로 커밋 그래프를 연결했다는 뜻이다.
**그 브랜치의 모든 옛 파일을 현재 실행 디렉터리에 덮어썼다는 뜻은 아니다.**
모든 tip이 HEAD의 조상이므로 통합 branch만 clone해도 원본 트리를 완전히 복원할 수 있다.
일반 merge와 이력 merge의 이유를 각각의 merge commit 메시지에도 남겼다. squash/rebase/force-push는 하지 않았다.

## 파일과 실행 경로의 판단

- `HW/`, STM32 `Core/`, Drivers, `.ioc`, linker script: main 우선. 9월 14일 yaw feedback off,
  압력센서·착지 개선 및 실제 보정값을 보존한다. 소스가 오래된 RL branch의 값을 역으로 덮지 않는다.
- 현재 `mjx/`: main의 path-v6·hybrid curriculum·영상 처리에 adaptive의 lookahead,
  free-Wave 선택·stance recovery·wall contact 분류·bounded foot retry·viewer HUD를 추가한다.
  신규 source hash/scheduler 계약도 함께 가져온다.
- `SW/Jetson/`: 최신 main 기반. 검사에서 발견한 `HardwareGeometry._config` 누락을 수정해
  공유 경로 검사기의 기본 `stair_clearance_extra=0.0`을 명시한다. 실제 정책/센서 연결 완료를 의미하지 않는다.
- `SW/mjx/`, `isaaclab_hexapod/`, 초기 MJX 학습 노트·tutorial: 과거 cartesian snapshot에서
  관련 asset·설정과 함께 복원한다. 오래된 18-D/Isaac 계약을 현재 24-D 배포로 오인하지 않게 한다.
  현재 코드의 기존 18-D replay와 golden artifact는 유지한다.
- v5 `resume_stair5_best.sh`: 원본 branch 전체와 같이 보존한다. 이 스크립트에는 당시 PC의
  `/home/huro/...` 경로가 있으므로 재현할 PC의 checkpoint 위치를 확인해야 한다.
  현재 v6 preset 두 개는 개인 절대경로 대신 `--checkpoint` 또는 `HEXAPOD_CHECKPOINT`를 받는다.
- 프로젝트 사진·PDF·발표 자료: 최신 main 파일을 보존한다.
- `.build/`, `SW/STM32/workspace/Hexapod/Debug/`, `SW/STM32/workspace/.metadata/`:
  재생성 가능한 빌드·개인 IDE 캐시 1,018개는 현재 추적에서 제외하고 `.gitignore`로 보호한다.
  이전 바이너리/로그는 main snapshot과 Git 이력에 남는다. 저장소 과거 이력을 다시 쓰지 않았다.

## 발전 단계

1. STM32 classical Tripod·접촉·캘리브레이션과 Wave gait.
2. Cartesian residual 및 18-D MJX 학습·Isaac Lab 포팅 실험.
3. main 위 선별 adaptive 통합: 24-D geometry residual v4와 128-byte adaptive codec.
4. elevation-grid CNN v5·완주 보상·RC profile·checkpoint 규격 검사.
5. path-v6 bottleneck geometry·명시적 v5 이전·평가 영상과 hybrid terrain curriculum.
6. adaptive의 phase 경계 free-Wave·stance 복구·wall contact/한 발 retract retry.
7. main의 최신 실기 센서·착지 개선을 유지한 현재 RL 통합과 단일 CLI.

각 단계 문서는 당시의 분석·검증 기록이므로 과거의 “최신” 표현이 남아 있다.
현재 기준은 [UNIFIED_WORKFLOW.md](UNIFIED_WORKFLOW.md)와 코드의 계약 상수다.

## 검증

검증 결과와 실행 환경은 [INTEGRATION_VALIDATION.md](INTEGRATION_VALIDATION.md)에 기록한다.
`./hexapod verify-history`는 모든 원본 tip의 조상 관계를 실제 Git으로 확인한다.
`./hexapod history <id> <경로>`는 브랜치 이름 변경·삭제와 무관하게 고정 SHA를 복원한다.
`./hexapod bundle`은 전체 도달 가능 이력을 한 파일로 모으며 로컬 학습 데이터는 포함하지 않는다.

향후 업데이트는 새 branch tip과 merge-base를 다시 검토해야 한다. 이 manifest는 오늘의 snapshot이며
새 원격 변경을 무조건 자동 merge하거나 checkpoint 검사를 우회하는 도구가 아니다.
