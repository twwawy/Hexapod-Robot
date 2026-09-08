> 현재 권장: **v6 수정 + 기존 계단 가중치**는
> `bash /home/huro/Hexapod-Robot-integration/scripts/resume_stair5_path_v6.sh`를 사용한다.
> 아래의 v5 전용 재개/이전 미지원 설명은 이전 상태다. `--migrate-path-v6`가
> d0370b3의 terrain v5 source 전체 hash를 확인한 뒤 명시적으로 이전한다.
> 이전 가중치의 원본 파일은 변경하지 않는다. 실행/학습 검증은 아직 하지 않았다.

# 계단 경로 여유 높이 기반 보정

## 변경 내용

발끝 경로의 최대 지형 높이만 사용하는 것에서, 실제 `planned_swing()`의 21개
progress sample에 대해 발바닥과 관측 지형 사이의 최소 수직 여유 높이를 추가로 계산한다.
최소 여유가 발생한 progress 및 출발 발끝으로부터의 수평 거리도 보관한다.
정확한 이륙·착지는 기존처럼 검사에서 제외한다(progress > .05, < .95).

각 25개 착지 후보에 fixed-size 네 가지 궤적을 검사한다.

| 번호 | 궤적 |
|---|---|
| 0 | Geometry + RL 원래 요청 |
| 1 | Apex를 최대 .10 앞당기고 XY transfer를 최대 .10 늦춤 |
| 2 | 부족한 여유 높이를 기준으로 clearance 증가 |
| 3 | Timing 및 clearance 함께 보정 |

추가 높이 후보는 `clip(1.5 * max(.006 - requested_margin, 0), .015, .06)` m다.
최종 clearance는 기존 .18 m 상한을 유지한다. 이 추가 높이는 geometry safety projection이며
RL residual 범위를 늘린 것이 아니다. .006 m는 보정 후보 생성 기준이고 실제 통과 조건은
기존처럼 관측 경로 최소 여유 .002 m 이상 및 required clearance 조건이다.

원래 요청이 안전하면 그대로 선택한다. 실패한 경우 IK·joint/workspace margin·경로 관측률·충돌
조건을 모두 통과하는 보정안 중 높이와 timing 변경 비용이 작은 것을 선택한다.
모두 실패하면 원래 요청을 rejected 상태로 남긴다. Unknown을 관측된 안전 지형으로 채우지 않는다.

## 보폭과의 연결

기존 stride bank의 각 보폭·착지 후보가 이 검사를 사용한다. 따라서 큰 보폭의 궤적은 실패하고
작은 보폭의 보정 궤적은 통과하면 supervisor의 feasible stride set에 차이가 생긴다.
RL은 기존 stride 선호와 landing XY residual을 계속 출력한다. 새로운 보폭 action이나
임의로 더 큰 XY 권한을 추가하지 않는다. 조합 전체의 전역 최적해를 보장하는 탐색은 아니다.

Wide search와 local residual refinement 모두 같은 evaluator를 쓴다.
선택한 clearance/apex/transfer는 기존 proposal→swing-entry latch 경로로 전달한다.
현재 active swing 도중에 다시 계획해서 목표를 바꾸는 기능은 추가하지 않았다.

## 관측 및 checkpoint

Candidate feature 28개에 다음 6개를 추가한다.

- 선택 궤적의 최소 여유 높이
- 여유가 최소인 progress
- 그 지점까지의 수평 거리
- 내부 경로 관측 여부
- 추가 clearance
- Timing 보정 여부

24-D action v4, elevation CNN 구조는 유지하지만 vector 입력 크기가 바뀐다.
Actor 7890 → 8790, critic 8205 → 9105.
Observation contract는 `adaptive_hybrid_elevation_grid24x24x6_path_v6`이며 metadata에
`sampled_foot_bottleneck_projection_v1`과 네 가지 궤적 variant를 기록한다.
**이전 checkpoint를 자동 restore하지 않는다. 새 run으로 시작한다.**
실행 중인 pinned source run에는 변경이 적용되지 않는다.

## 진단

Viewer 콘솔은 proposal variant, 최소 여유 높이(m), bottleneck progress/거리,
추가 높이를 출력한다. 저장하는 foothold plan/diagnostics에는 원래 요청의 margin/progress도 포함한다.

W&B 학습 metrics:

- `path/proposal_repair_fraction`
- `path/proposal_height_correction_m`
- `path/proposal_min_margin_m`
- `path/proposal_observed_fraction`

위 값은 다음 계획의 진단이며 실제 접촉이나 active swing의 측정값이 아니다.
Brax episode 누적값은 순간 거리/비율과 구분해야 한다. 특히 누적 margin을 episode의
최소 margin으로 읽지 않는다. 정확한 후보 값은 viewer 저장 plan을 사용한다.
기존 완주 보상·안전/효율 비용·cycle별 영상 방식은 유지한다.

## 한계 및 사용자 검사

발끝 구 footprint의 이산 샘플 검사다. 다리 링크 전체 충돌, 샘플 사이 충돌,
지도 오차 및 실제 접촉을 완전히 보장하지 않는다. 네 개 궤적 검사로 JIT/step 비용이 늘며
실제 처리량은 측정하지 않았다. STM32 궤적식 및 하드웨어 코드는 이번 변경에서 수정하지 않았다.

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
python -m unittest mjx.tests.test_path_bottleneck mjx.tests.test_adaptive_hybrid
```

이 검사는 작성만 했고 실행하지 않았다. 문법 AST 및 diff만 확인했다.
학습·시뮬레이션·보드 검증은 사용자가 수행한다. 시작 명령은 README를 사용하되
새 이름 `--run-name forward-gt-path-v6`로 실행하고 `--restore`를 넣지 않는다.
