# Hexapod 문서 안내

최종 갱신: 2026-09-06. 현재 기본 뷰어는 **펌웨어 보행 + stage31 residual**이다.
LiDAR 착지 후보를 절대 경로로 넣는 이전 전환 방식은 제거했다.

| 문서 | 용도·상태 |
|---|---|
| [루트 README](../README.md) | 시작 명령, 환경, 조작, v3/v4 경로 구분 |
| [뷰어 사용법](HEXAPOD_FOOTHOLD_PREVIEW_USAGE.md) | 현재 실행 구성, scale 0 비교, 사용자 확인과 진단 저장 |
| [LiDAR/residual 설계](HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md) | 최신 결정, GT 정답과 actor 입력 분리, 미구현 학습 계획 |
| [업데이트 기록](HEXAPOD_UPDATE_2026-09-06.md) | 이번 수정과 이전 미커밋 소스·문서·모델 반영 범위 |
| [MJX 학습 설계](../mjx/RL_DESIGN.md) | 현재 루트 v4/18-D·146-D, curriculum 0~16, reward/checkpoint |
| [펌웨어 설명](../mjx/FIRMWARE_BASE.md) | nominal gait·IK·servo 기반 |
| [stage31 패키지](../mjx/policies/progress-v2-stage31-level6/README.md) | 기록된 v3 checkpoint 출처와 독립 replay |
| [Isaac README](../isaaclab_hexapod/README.md) | Isaac 모델·센서·학습 scaffold 실행 |
| [Isaac 이식 기록](../isaaclab_hexapod/PORT_RESULT_AND_USAGE.md) | 기존 생성/검사 기록과 남은 작업 |
| [MJX→Isaac 계획](HEXAPOD_MJX_TO_ISAACLAB_PORT_PLAN.md) | 이식 계약과 단계별 계획 |
| [이전 perceptive 설계](HEXAPOD_PERCEPTIVE_RESIDUAL_ISAACLAB_PLAN.md) | LiDAR+Depth 및 asymmetric actor/critic 참고; 최신 viewer 계약과 구분 |
| [학습 스터디 노트](Hexapod_MJX_Obsidian_Study_Vault.md) | 구현 이해용 노트; 실행 옵션은 현재 코드/사용법 우선 |
| [레거시 residual 설계](RESIDUAL_RL.md) | 과거 `SW/mjx/` 계약 참고 |

기존 JSON 검사 보고서와 문서의 과거 성공 기록은 당시 산출물에 대한 기록이다.
이번 통합 변경의 GUI·정책 추론·보행·학습 검증은 사용자에게 맡겼으며 다시 실행하지 않았다.
