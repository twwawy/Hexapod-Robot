# Adaptive MJX: 다음 phase 예측과 자유 순서 Wave

2026-09-09 구현. **실제 rollout·PPO·실기 검증은 수행하지 않았다.**
적용 경로는 `codex/adaptive-hybrid-rl-integration`의 adaptive MJX다.
STM32, manual gait, legacy 18-D, 기구 치수/좌표계는 이번 변경 대상이 아니다.

## 확인된 문제와 변경

기존 checkpoint 진단에서는 RB가 stance인 상태에서도 IK/reach 검사를
통과하지 못했다. 다른 발을 먼저 들어도 RB가 지지 다리로 남기 때문에
고정 Wave 순서만 해제하는 것으로는 이 상태를 해결할 수 없다.

1. **다음 Tripod 예측**: 현재 phase의 6개 보폭 각각에 대해 착지 후
   발 위치·몸통 전진을 예측하고 반대 그룹의 후보/IK/path/support를 검사한다.
   충분히 관측된 다음 phase가 불가능하면 해당 보폭을 제외한다.
   기존에는 normal 보폭의 다음 phase를 주로 Wave→Tripod 복귀에만 사용했다.
   예측의 unknown만으로 known-infeasible 또는 Wave 전환을 만들지 않는다.
2. **자유 순서 Wave**: 6개 발 × 보폭 후보를 비교한다. 기존 순서 배열
   `RF LB RM LF RB LM`은 후보 인덱스용이며 실제 실행 순서를 강제하지 않는다.
   안전한 후보 중 지지 여유와 마지막 착지 후 경과 phase 수로 선택한다.
   나이 점수는 편중을 줄이지만 unsafe 발을 강제로 선택하지 않는다.
   시작 시 발 번호/목표/clearance/apex/transfer를 고정한다.
   최초 6회 startup의 half stance rate는 발 번호 대신 완료 횟수로 관리한다.
3. **제자리 발 재배치**: 전진 가능한 Wave가 하나도 없을 때에만 stride=0
   후보를 허용한다. 한 발의 안전한 foothold를 다시 잡고 몸통 전진은 0으로 둔다.
   Tripod UNKNOWN→Wave 전환 금지는 그대로다.
4. **전발 접촉 자세 복구**: planner가 HOLD이고 swing/recontact/fault가 없을 때
   13개 고정 후보(현상 유지, 축별 ±2cm translation 또는 ±3° roll/pitch)를 검사한다.
   전발 지지 자세를 21점으로 샘플링하여 IK/reach와 지지 여유 2cm를 확인한다.
   가장 작은 reach margin을 1mm 이상 개선하는 후보만 실행한다.
   2초 quintic 진행률로 기존 foot memory에서 검사한 목표로 이동하며,
   매 tick IK/reach를 다시 검사한다. phase당 한 번만 시도하여 누적 이동을 막는다.
   접촉 상실, fault, command 해제 시 중단한다. 복구 중 새 swing은 시작하지 않는다.

## 책임과 한계

- RL action은 기존 24-D v4, actor/critic 배열은 path-v6 8790/9105 그대로다.
- landing Z는 terrain 소유다. 제자리 재배치에도 동일한 terrain 후보 검사를 쓴다.
- 자세 복구는 전발 접촉 상태의 controller pre-posture foot-frame 이동이다.
  이를 policy landing-Z residual로 해석하지 않는다.
- **이미 시작 자세가 IK 밖이면 복구 경로도 거부된다.** 제한을 완화하거나
  invalid 시작점을 건너뛰지 않는다. 이번 변경이 진단된 RB 막힘을 해결했다고
  주장하지 않는다. 예방 예측으로 그 상태를 피하는지 사용자가 확인해야 한다.
- 예측은 현재 자세 기반 근사이며 접촉 동역학·미끄러짐을 보장하지 않는다.
  recovery support 계산의 CoM은 controller 원점 근사다. 전체 몸통 충돌 검사는 아니다.
- 다음 phase 검사는 classical reference 기준이다. RL 이후 실제 경로는 기존
  projection/controller gate가 검사한다. 미래 자세 오차는 남는다.
- 고정 shape/JAX 경로를 유지했지만 다음 phase 및 Wave 검사량이 늘었다.
  compile 시간·GPU 사용량은 측정하지 않았다.
- 기구/좌표계 검토 및 STM32 adaptive 자유 순서 Wave 포팅은 보류 상태다.
  이 MJX 변경을 기존 STM32와 동등하다고 취급하면 안 된다.

## 진단

viewer 콘솔:

- `lookahead_blocked`: 다음 phase 예측 때문에 제외된 6개 Tripod 보폭
- `wave_candidates_RF_LB_RM_LF_RB_LM`: 발별 실행 가능 여부
- `wave_choice_phase`: 선택한 Wave 후보 인덱스
- `recenter`, `time`: 전발 자세 복구 진행 여부/시간

학습 metrics:
`lookahead/blocked_strides`, `wave/available_legs`, `recenter/active_s`.
기존 planner HOLD, contact wait, recontact exhausted/IK blocked도 함께 확인한다.

## 사용자가 실행할 검증

```bash
cd /home/huro/Hexapod-Robot-integration
source /home/huro/.venvs/hexapod-mjx/bin/activate
python -m unittest mjx.tests.test_adaptive_hybrid
bash scripts/view_foothold_planner.sh \
  --controller adaptive --terrain steps --perception oracle \
  --gait-mode hybrid --speed 0.04 --stage0
```

oracle에서 먼저 zero-action의 진행/전환/복구를 확인한 뒤 `--perception lidar`로 비교한다.
발을 고른 뒤 mid-swing에 다른 발로 바뀌지 않는지, 복구가 접촉을 잃으면
멈추는지, 제자리 발 재배치 후 실제 전진 가능한 후보가 생기는지 확인한다.
테스트에는 boundary latch, startup 완료 횟수, zero stride 선택,
invalid recovery 시작점 거부와 복구 경로 검사 항목을 추가했다.

## 기존 checkpoint

소스가 바뀌었으므로 일반 `--restore`의 source mismatch를 우회하지 않는다.
기존 `--migrate-path-v6`에 **53bab78의 전체 source manifest** 확인을 추가했다.
기존 허용 대상 d0370b3(v5), 585bee2(v6)도 유지한다.
새 optimizer를 사용하는 명시적 warm start이며 학습 재현/동작 호환 보장이 아니다.

아래 checkpoint를 사용할 경우 기존 curriculum 명령에 다음 두 인자를 붙인다.
경로는 반드시 한 줄로 붙여 넣는다.

```bash
--restore /home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum/hybrid-stair5-v6-clearance2cm/02_hybrid-stair5_try11/checkpoints/000000204800 \
--migrate-path-v6
```

직전 검사 이후 생성된 다른 revision의 checkpoint는 자동으로 허용하지 않는다.
현재 checkpoint replay는 기록된 원래 소스에서 실행해야 하며,
새 controller 비교는 명시적 warm start와 별도 run 이름으로 구분한다.
