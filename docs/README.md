# Hexapod 문서 안내

최종 갱신: 2026-09-06. 새 작업 기준은 **adaptive 24-D hybrid 보행**이다.
`--controller adaptive`로 선택하며, 옵션을 생략하는 기본 뷰어는 기존 **stage31 18-D 비교 모드**다.
두 모드는 후보의 역할·미관측 대응·checkpoint 계약이 다르다.

## 현재 adaptive 설계·실행

| 문서 | 용도·상태 |
|---|---|
| [루트 README](../README.md) | 프로젝트 구성·구현 요약·실행·환경·학습·레거시 안내 |
| [Hybrid 구현 전 분석](HEXAPOD_HYBRID_GAIT_ANALYSIS.md) | 변경 전 action/observation·후보 gate·main Wave 대조 |
| [Adaptive 사용 가이드](HEXAPOD_MJX_ADAPTIVE_GAIT_USAGE.md) | 24-D 계약·oracle·후보 진단·Tripod/Wave·학습 명령·제한 |
| [MJX 지형 적응 보행 학습 계획](HEXAPOD_MJX_ADAPTIVE_GAIT_LEARNING_PLAN.md) | 구현된 geometry/residual/supervisor 구조와 Stage 0–3·MJX→Isaac 순서 |

## 기존 모드·업데이트 기록

| 문서 | 용도·상태 |
|---|---|
| [기존 뷰어 사용법](HEXAPOD_FOOTHOLD_PREVIEW_USAGE.md) | stage31 실행·scale 0 비교·사용자 확인·진단 저장 |
| [LiDAR/residual 설계](HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md) | stage31 LiDAR 입력 교체와 후속 학습의 설계 배경 |
| [업데이트 기록](HEXAPOD_UPDATE_2026-09-06.md) | 당시 viewer·소스·문서·모델 변경 기록; 최신 adaptive 계약은 위 가이드 참고 |
| [MJX 학습 설계](../mjx/RL_DESIGN.md) | 현재 루트 v4/18-D·146-D, curriculum 0~16, reward/checkpoint |
| [펌웨어 설명](../mjx/FIRMWARE_BASE.md) | nominal gait·IK·servo 기반 |
| [stage31 패키지](../mjx/policies/progress-v2-stage31-level6/README.md) | 기록된 v3 checkpoint 출처와 독립 replay |
| [레거시 residual 설계](RESIDUAL_RL.md) | 과거 `SW/mjx/` 계약 참고 |

## Isaac Lab·참고 자료

| 문서 | 용도·상태 |
|---|---|
| [Isaac README](../isaaclab_hexapod/README.md) | Isaac 모델·센서·학습 scaffold 실행 |
| [Isaac 이식 기록](../isaaclab_hexapod/PORT_RESULT_AND_USAGE.md) | 기존 생성/검사 기록과 남은 작업 |
| [MJX→Isaac 계획](HEXAPOD_MJX_TO_ISAACLAB_PORT_PLAN.md) | 이식 계약과 단계별 계획 |
| [이전 perceptive 설계](HEXAPOD_PERCEPTIVE_RESIDUAL_ISAACLAB_PLAN.md) | LiDAR+Depth 및 asymmetric actor/critic 참고; 최신 viewer 계약과 구분 |
| [학습 스터디 노트](Hexapod_MJX_Obsidian_Study_Vault.md) | 구현 이해용 노트; 실행 옵션은 현재 코드/사용법 우선 |

기존 JSON 검사 보고서와 문서의 과거 성공 기록은 당시 산출물에 대한 기록이다.
이번 통합 변경의 GUI·정책 추론·보행·학습 검증은 사용자에게 맡겼으며 다시 실행하지 않았다.
