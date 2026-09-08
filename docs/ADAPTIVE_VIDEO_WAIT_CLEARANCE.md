# 평가 영상 대기와 계단 clearance 조정

## 대기 진단

409600-step 영상/manifest는 생성됐다. W&B 내부 로그에는 이후에도 HTTP 200 응답이 있었다.
따라서 마지막 `Adding directory... Done`만으로 업로드 교착을 확정할 수 없다.
부모가 `process.stdout`을 기다리던 KeyboardInterrupt는 사용자의 Ctrl+C 중단이다.
PPO 연산인지 업로드 처리인지 그 시점의 child stack은 확보하지 않았으므로 정확한 병목은 미확정이다.

실제 확인한 중복 작업을 줄였다.

- 현재 정책은 매 평가 새로 렌더링한다. Best가 갱신된 경우 해당 영상을 best로도 게시하며 중복 rollout하지 않는다.
- 모든 평가의 현재 checkpoint는 current-policy artifact로 저장한다. Best-policy artifact는 best 갱신 시에만 추가한다.
- `EVAL VIDEO START`, `CACHE`, `RETURN`으로 callback의 진입/복귀를 표시한다.
- 부모는 30초마다 `WORKER WAIT`를 표시한다. 이는 프로세스 생존 표시이며 PPO 진행 보장이 아니다.
- Ctrl+C 시 부모가 child에 terminate하고 10초 안에 종료하지 않으면 kill하여 방치하지 않는다.

새 best 영상 생성은 여전히 동기식이다. 렌더링 중 PPO는 기다린다.
백그라운드 GPU renderer나 처리량 개선을 검증했다고 주장하지 않는다.

## 발 높이

`--stair-clearance-extra 0.02`는 착지 표면이 출발 발바닥보다 2.5cm 이상 높은 경우,
geometry baseline 및 최소 clearance에 2cm를 더한다. 평지에는 추가하지 않는다.
RL residual 범위, body pitch/height, 보폭 설정은 유지한다.
clearance 18cm 상한과 IK/path 검사는 유지하므로 도달 불가 궤적을 강제하지 않는다.
추가 높이가 IK 여유를 줄여 후보를 탈락시킬 수도 있다. 실제 개선 여부는 사용자 검증 대상이다.
CLI 범위는 0~4cm이며 기본값은 0이다. 설정은 checkpoint config에 기록된다.

## 중단된 v6 best로 재개

```bash
bash /home/huro/Hexapod-Robot-integration/scripts/resume_stair5_v6_clearance.sh
```

`stair5-path-v6-warmstart/02_tripod-stair5_try00/checkpoints/000000204800`을 사용한다.
`--migrate-path-v6`는 기록된 585bee2 v6의 전체 source hash를 확인한 뒤 이번 geometry 변경을
명시적으로 허용한다. v6→v6는 입력 차원/가중치 변환 없이 이어받는다. Optimizer는 새로 시작한다.
기존 d0370b3 v5→v6 명시적 이전도 유지한다. 임의 source hash bypass는 허용하지 않는다.

학습·시뮬레이션·영상 렌더링·변환 실행 검증은 하지 않았다. AST/shell 문법 및
기존 checkpoint의 source hash 읽기만 확인했다.

재개 스크립트는 평가 9회·현재 정책 영상 20초·baseline 비교 0으로 설정한다. 학습량 800000과 episode 8000은 유지한다.

현재 재개 preset은 `--profile hybrid --start-index 2`로 Hybrid stair5부터 진행한다. 별도 Tripod/Wave 학습 단계는 없다.
