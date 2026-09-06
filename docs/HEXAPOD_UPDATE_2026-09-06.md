# 2026-09-06 업데이트 정리

대상 브랜치: `codex/cartesian-residual-rl`.
이번 작업은 보행 제어 수정, README/docs 정리와 기존 미커밋 변경의 저장소 반영이다.

## 보행 뷰어 수정

사용자가 비어 있는 지도에서는 걷지만 관측된 착지점 진입 시 IK가 꼬이는 현상을 보고했다.
이전 어댑터는 geometric world 경로를 nominal 목표 대신 넣고, 착지 뒤 corrected foot memory를
펌웨어에 되돌렸다. 두 궤적/접촉 처리의 충돌 가능성이 있어 해당 경로와 다리별 takeover를 제거했다.
실행 로그로 IK 실패의 직접 원인을 확정하거나 수정 후 등반 성공을 확인한 것은 아니다.

현재 구조는 다음과 같다.

- 원래 격리된 v3 펌웨어 step: nominal gait·접촉·posture·foot memory·residual 합성·IK·관절 속도 제한.
- 어댑터: 관측 비율과 `--residual-scale`로 policy action 크기만 조절. 0.5초 gain smoothing.
- 높이 15개 샘플이 모두 미관측이면 action/filter=0, 기본 제어기로 계속 보행.
- geometric 후보는 참고 표시이며 후보 실패로 기본 동역학 gait를 멈추지 않음.
- GT 기반 지형 보정은 끄고 LiDAR만 actor 높이 입력에 사용. GT는 물리·환경 평가·별도 진단으로 분리.
- HUD/JSON에 gain, IK valid, residual IK valid, reach limited와 LiDAR/GT 비교를 추가.
- P로 동일 XY의 LiDAR/GT 높이·valid·age·입력 pose/시각을 `latest_lidar_gt_pair.npz`에 저장.

## 함께 반영한 기존 미커밋 변경

아래는 작업 폴더에 이미 존재하던 변경을 이번 요청으로 함께 반영한 목록이다.
이번 뷰어 수정에서 새로 학습하거나 검증했다는 의미가 아니다.

| 범위 | 반영 내용 |
|---|---|
| CAD/URDF | MID-360 CAD 장착 반전 및 visual/inertia/fixed-chain 수정 |
| MJX 모델/asset export | CAD visual과 primitive collider를 구분하는 export·scene 준비 변경 |
| MJX 제어 계약 | v4의 swing X/Y·stance Z 축별 최대 100 mm residual 요청, 146-D observation |
| MJX 지형/학습 | 6.5/8 cm 중간 계단 포함 level 0~16, progress/pitch/swing 보조·보상·승급/체크포인트 관련 수정 |
| MJX 도구/자료 | benchmark 도구, teacher manifest, golden contract 자료와 기존 테스트 소스 변경 |
| Isaac asset | CAD mesh를 보존한 USD 및 생성/검사 스크립트·기록 |
| Isaac 제어·센서 | Torch controller, joint contract, MID-360 설정, terrain encoder, MJX curriculum 지형 |
| Isaac 실행/학습 | realtime/model viewer, MJX handoff 동기화, RSL-RL 실행·staged 스크립트와 구성 |
| 문서 | root README, docs index/설계/사용법, MJX와 Isaac 안내의 최신 상태·버전 구분 |

기본 viewer는 **stage31 v3**를 계속 사용한다. 루트 개발 코드와 Isaac의 **v4**를 그 checkpoint에
자동 적용하지 않는다. Isaac `data/training/latest_mjx_training.json`은 기존 run 기록과 호환성/안전 gate를
담은 handoff이며, 이번 코드 변경 후의 신규 학습 결과가 아니다.

measured LiDAR TF(215 mm, 전방 45°)와 URDF CAD 장착 chain은 별도 출처다.
이번 CAD 변경을 measured TF와 같은 값이라고 해석하지 않는다.

## 파일 관리와 검증 상태

소스·설명·재현에 필요한 모델/메타데이터를 반영한다. 실행 로그, Hydra 출력, 영상·중간 checkpoint는
`logs/`, `outputs/`, `isaaclab_hexapod/outputs/`에서 로컬 실행 산출물로 관리한다.
별도로 선택해 패키징한 stage31 checkpoint는 `mjx/policies/`에서 계속 추적한다.

이번 작업에서 수행하는 확인은 소스 문법/문서 링크/변경 diff 수준이다.
MuJoCo GUI, policy inference, 물리 보행, Isaac 시뮬레이션, 학습 및 테스트 스위트는 실행하지 않는다.
기존 validation JSON의 날짜·결과는 과거 기록으로 보존하며 최신 통합 검증으로 갱신하지 않는다.

사용자 실행/확인 순서는 [뷰어 안내](HEXAPOD_FOOTHOLD_PREVIEW_USAGE.md)에 있다.
향후 GT label로 인식 모듈을 학습하고 sensor student를 distill/fine-tune하는 계획은
[LiDAR/residual 설계](HEXAPOD_LIDAR_FOOTHOLD_RESIDUAL_PLAN.md)에 분리했다.
